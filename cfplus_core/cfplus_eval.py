"""Module for training and evaluating CF+ models."""
from typing import Any, Dict, List, Tuple, Union
from copy import deepcopy
from pathlib import Path
import pickle as pkl
import random
import numpy as np
import torch

from utils import constants
from utils import utils_data
from utils import utils_eval
from cf_core.cf_model import CFModel, add_user_to_model
from cf_core import utils_cf
from cfplus_core.fm_model import FMRegressor
from cfplus_core import utils_cfplus


def calculate_fm_distances(
    struct_dict: Dict[str, Any],
    user_observation_dict: Dict[str, Any],
    hidden_dimension: int=30,
    num_iter: int=1000,
    init_lr: float=0.03,
    init_stdev: float=0.1,
):
    """Train an FM model from user observations and use it to generate obj-loc scores.

    Scores are generated for each unplaced objects with each surface and returned
    as a matrix.
    """
    if struct_dict["user_id"] not in user_observation_dict:
        raise ValueError("Observations from test user not in the input dictionary.")
    unplaced_object_list = list(obj.object_id for obj in struct_dict["partial"][-1].objects_on_surface)
    train_vectors, train_labels, vectorizer, surface_metadata_list = utils_cfplus.generate_vectors_from_observations(
        user_observation_dict
    )
    surface_name_list = [x["name"] for x in surface_metadata_list]
    fm_regressor = FMRegressor(
        hidden_dimension=hidden_dimension,
        num_iter=num_iter,
        init_stdev=init_stdev,
        learn_rate=init_lr,
    )
    prediction_matrix = np.inf*np.ones((len(unplaced_object_list), len(surface_metadata_list)))
    object_surface_pairs = []
    matrix_indices = []
    for i, obj in enumerate(unplaced_object_list):
        for j, surf_dict in enumerate(surface_metadata_list):
            if f"object_id={obj}" in vectorizer.feature_names_:
                object_surface_pairs.append(
                    {
                        "user_id": struct_dict["user_id"],
                        "object_id": obj,
                        "surface_name": surf_dict["name"],
                        "surface_type": surf_dict["surface_type"],
                    }
                )
            else:
                object_surface_pairs.append(None)
            matrix_indices.append((i, j))
    
    valid_indices = [i for i, x in enumerate(object_surface_pairs) if x is not None]
    if not valid_indices:  # If all objects are new, return an empty matrix.
        return prediction_matrix, unplaced_object_list, surface_name_list
    test_vectors = vectorizer.transform([
        object_surface_pairs[i] for i in valid_indices
    ])
    test_predictions = fm_regressor.fit_predict(
        train_vectors, train_labels, test_vectors, shuffle=True
    )
    for l, (pair_dict, pair_coords) in enumerate(zip(object_surface_pairs, matrix_indices)):
        if pair_dict is not None:
            pred_index = valid_indices.index(l)
            pred = test_predictions[pred_index]
            prediction_matrix[pair_coords] = pred
    return prediction_matrix, unplaced_object_list, surface_name_list


def _calculate_cf_object_surface(
    query_object_id: str,
    surface_objects: List[str],
    user_id: str,
    ranker: utils_cf.CFRanker,
):
    """Calculate CF distances for a single object on a partial scene."""
    if not surface_objects:  # If the surface is empty, return mean ranking.
        return np.sum(ranker.ranking_matrix) / ranker.ranking_matrix.size
    else:  # Return minimum ranking of the objects on the surface.
        rank_array = []
        for obj in surface_objects:
            try:
                rank_array.append(ranker(query_object_id, obj, user_id=user_id))
            except KeyError:
                rank_array.append(np.inf)
        return min(rank_array)

def calculate_cf_distances(
    struct_dict: Dict[str, Any],
    cf_model: CFModel,
    hyperparams: Dict[str, Union[int, float]],
    user_list_train: List[str],
    object_combinations: List[Tuple[str, str]],
):
    """Calculate CF distances for unplaced objects on a partial scene."""
    ranking_matrix = cf_model.forward().detach().cpu().numpy()
    if struct_dict["user_id"] in user_list_train:
        ranker = utils_cf.CFRanker(
            combinations=object_combinations,
            ranking_matrix=ranking_matrix,
            user_list=user_list_train,
            object_ID_list_semantic=None,
            semantic_similarity_matrix=None,
        )
    else:
        matrix_test, _, non_neg_indices_test, _ = utils_cf.batch_to_ranking_matrix(
            batch={"dummy_scene": struct_dict},
            object_combinations=object_combinations
        )
        assert matrix_test.shape[1] == 1 and len(non_neg_indices_test) == 2
        new_model = add_user_to_model(
            model=cf_model,
            new_ratings=matrix_test,
            non_neg_indices=non_neg_indices_test,
            learning_rate=hyperparams["learning_rate"],
            convergence_threshold=hyperparams["convergence_threshold"],
        )
        new_rank_matrix = new_model.forward().detach().cpu().numpy()
        ranker = utils_cf.CFRanker(
            combinations=object_combinations,
            ranking_matrix=new_rank_matrix,
            user_list=user_list_train + [struct_dict["user_id"]],
            object_ID_list_semantic=None,
            semantic_similarity_matrix=None,
        )
    partial_scene = struct_dict["partial"]
    assert partial_scene[-1].surface_type == constants.SURFACE_TYPES.NOT_PLACED
    object_id_per_surface = {
        surf.name: [o.object_id for o in surf.objects_on_surface]
        for surf in partial_scene[:-1]
    }
    surface_name_list = list(surf.name for surf in partial_scene[:-1])
    unplaced_object_list = list(o.object_id for o in partial_scene[-1].objects_on_surface)
    prediction_matrix = np.inf*np.ones((len(unplaced_object_list), len(surface_name_list)))
    for i, obj_id in enumerate(unplaced_object_list):
        for j, surf_name in enumerate(surface_name_list):
            if all(obj_id not in comb for comb in object_combinations):
                continue
            prediction_matrix[i, j] = _calculate_cf_object_surface(
                obj_id,
                object_id_per_surface[surf_name],
                struct_dict["user_id"],
                ranker
            )
    return prediction_matrix, unplaced_object_list, surface_name_list


def calculate_rank(
    struct_dict: Dict[str, Any],
    user_list_train: List[str],
    train_observation_dict: Dict[str, Any],
    cf_object_combinations: List[Tuple[str, str]],
    cf_model: CFModel,
    cf_hyperparams: Dict[str, Union[int, float]],
    fm_hidden_dimension: int=30,
    fm_num_iter: int=1000,
    fm_init_lr: float=0.03,
    fm_init_stdev: float=0.1,
):
    """Calculate CF+ object-surface distances for unplaced objects on a partial scene."""
    cf_distances, unplaced_object_list, surface_name_list = calculate_cf_distances(
        struct_dict=struct_dict,
        cf_model=cf_model,
        hyperparams=cf_hyperparams,
        user_list_train=user_list_train,
        object_combinations=cf_object_combinations,
    )
    fm_distances, unplaced_object_list_2, surface_name_list_2 = calculate_fm_distances(
        struct_dict=struct_dict,
        user_observation_dict=train_observation_dict,
        hidden_dimension=fm_hidden_dimension,
        num_iter=fm_num_iter,
        init_lr=fm_init_lr,
        init_stdev=fm_init_stdev,
    )
    assert len(unplaced_object_list) == len(unplaced_object_list_2)
    assert len(surface_name_list) == len(surface_name_list_2)
    if any(
        unplaced_object_list[i] != unplaced_object_list_2[i]
        for i in range(len(unplaced_object_list))
    ) or any(
        surface_name_list[i] != surface_name_list_2[i]
        for i in range(len(surface_name_list))
    ):
        raise NotImplementedError("Unplaced object list or surface name list mismatch has not been coded up.")
    no_rating_mask = np.bitwise_and(cf_distances > 1, fm_distances > 1)
    # Calculate harmonic mean of CF and FM distances.
    combined_distances = np.inf*np.ones_like(cf_distances)
    combined_distances[~no_rating_mask] = 2 / (1 / cf_distances[~no_rating_mask] + 1 / fm_distances[~no_rating_mask])
    assert np.all(combined_distances[no_rating_mask] > 1)
    return combined_distances, unplaced_object_list, surface_name_list, {
        "cf": cf_distances,
        "fm": fm_distances,
    }


def run_batch(
    cf_model: CFModel,
    hyperparams_cf: Dict[str, Union[int, float]],
    hyperparams_fm: Dict[str, Union[int, float]],
    train_batches: Dict[str, Any],
    eval_batches: Dict[str, Any],
    user_list_train: List[str],
    object_combinations: List[Tuple[str, str]],
):
    """Run a batch of evaluations on the CF+ model."""
    train_observation_dict = utils_cfplus.batch_to_observation_dict(train_batches)
    for key, struct_batch in eval_batches.items():
        extended_observation_dict = deepcopy(train_observation_dict)
        extended_observation_dict[struct_batch["user_id"]] = struct_batch["observed_arrangement_list"]
        combined_distances, unplaced_object_list, surface_name_list, dist_mat_dict = (
            calculate_rank(
                struct_dict=struct_batch,
                user_list_train=user_list_train,
                train_observation_dict=extended_observation_dict,
                cf_object_combinations=object_combinations,
                cf_model=cf_model,
                cf_hyperparams=hyperparams_cf,
                fm_hidden_dimension=hyperparams_fm["hidden_dimension"],
                fm_num_iter=hyperparams_fm["num_iter"],
                fm_init_lr=hyperparams_fm["init_lr"],
                fm_init_stdev=hyperparams_fm["init_stdev"],
            )
        )
        partial_scene = struct_batch["partial"]
        surface_name_assignments = []
        for i in range(len(unplaced_object_list)):
            if np.all(combined_distances[i] > 1):
                surface_name_assignments.append(random.choice(surface_name_list))
            else:
                surface_name_assignments.append(
                    surface_name_list[np.argmin(combined_distances[i])]
                )
        # Generate a new scene while only placing the unplaced objects.
        predicted_scene = deepcopy(partial_scene)
        surface_names = [surf.name for surf in predicted_scene]
        for object_id, assigned_surf in zip(
            unplaced_object_list,
            surface_name_assignments
        ):
            matching_unplaced_obj = [
                obj for obj in predicted_scene[-1].objects_on_surface
                if obj.object_id == object_id
            ]
            assert len(matching_unplaced_obj) >= 1, f"Object {object_id} not found in unplaced objects."
            matching_unplaced_obj = matching_unplaced_obj[0]

            surf_index = surface_names.index(assigned_surf)
            predicted_scene[-1].objects_on_surface.remove(matching_unplaced_obj)
            predicted_scene[surf_index].objects_on_surface.append(
                matching_unplaced_obj
            )
        
        goal_scene = struct_batch["goal"]
        yield {
            "scene_id": key,
            "unplaced_objects": unplaced_object_list,
            "surfaces_names": surface_name_list,
            "cf_distances": dist_mat_dict["cf"],
            "fm_distances": dist_mat_dict["fm"],
            "combined_distances": combined_distances,
            "partial_scene": partial_scene,
            "predicted_scene": predicted_scene,
            "goal_scene": goal_scene,
        }


def evaluate(
    checkpoint_folder: Path,
    hyperparams_cf: Dict[str, Union[int, float]],
    hyperparams_fm: Dict[str, Union[int, float]],
    train_batches: Dict[str, Any],
    eval_batches: Dict[str, Any],
):
    """Wrapper function around run_batch to calculate evaluation metrics."""
    result_array = []
    prediction_array = {}
    latent_embeddings = {}

    with open(checkpoint_folder / "train_params.pkl", "rb") as fp:
        cf_train_params = pkl.load(fp)
    cf_model = CFModel(
        hyperparams_cf["hidden_dimension"],
        len(cf_train_params["object_combinations"]),
        len(cf_train_params["user_list_train"]),
        hyperparams_cf["lambda_reg"],
        mode="train",
    ).double()
    cf_model.load_state_dict(torch.load(checkpoint_folder / "model.ckpt"))

    response_generator = run_batch(
        cf_model=cf_model,
        hyperparams_cf=hyperparams_cf,
        hyperparams_fm=hyperparams_fm,
        train_batches=train_batches,
        eval_batches=eval_batches,
        user_list_train=cf_train_params["user_list_train"],
        object_combinations=cf_train_params["object_combinations"],
    )

    for response in response_generator:
        key = response["scene_id"]
        edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
            response["predicted_scene"], response["goal_scene"]
        )
        unplaced_objects = response["partial_scene"][-1].objects_on_surface
        unplaced_object_classes = set(o.name for o in unplaced_objects)
        result_array.append(
            {
                "scene_id": key,
                "container_type": eval_batches[key]["container_type"],
                "household": eval_batches[key]["household"],
                "num_demonstrations": eval_batches[key]["num_demonstrations"],
                "num_removed_demonstrations": sum(
                    int(x) for x in eval_batches[key]["num_removed_demonstrations"]
                ),
                "num_misplaced_objects": len(unplaced_objects),
                "num_misplaced_object_classes": len(unplaced_object_classes),
                "edit_distance": edit_distance,
                "igo": igo,
                "ipc": ipc,
            }
        )
        prediction_array[key] = {
            "predicted_scene": utils_data.scene_to_json(response["predicted_scene"]),
            "goal_scene": utils_data.scene_to_json(response["goal_scene"]),
        }
        latent_embeddings[key] = {
            "unplaced_objects": response["unplaced_objects"],
            "surfaces_names": response["surfaces_names"],
            "cf_distances": response["cf_distances"],
            "fm_distances": response["fm_distances"],
            "combined_distances": response["combined_distances"],
        }  # TODO: remove if redundant.
    return result_array, prediction_array, latent_embeddings