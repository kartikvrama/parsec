"""CF evaluation functions."""

from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from copy import deepcopy
import pickle as pkl
import numpy as np
import torch
from cf_core import utils_cf
from cf_core import cf_model
from utils import constants
from utils import utils_data
from utils import utils_eval

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")


def run_batch(
    model: cf_model.CFModel,
    hyperparams: Dict[str, Union[int, float]],
    eval_batches: Dict[str, Any],
    user_list_train: List[str],
    object_combinations: List[Tuple[str, str]],
    object_ID_list_semantic: List[str],
    semantic_similarity_matrix: np.ndarray,
    object_id_dict: Dict[str, int],
):
    """Run CF model on a batch of evaluation data."""
    ranking_matrix = model.forward().detach().cpu().numpy()
    ranker_original = utils_cf.CFRanker(
        combinations=object_combinations,
        ranking_matrix=ranking_matrix,
        user_list=user_list_train,
        object_ID_list_semantic=object_ID_list_semantic,
        semantic_similarity_matrix=semantic_similarity_matrix,
    )

    for key, struct_batch in eval_batches.items():
        if struct_batch["user_id"] not in user_list_train:
            matrix_test, _, non_neg_indices_test, _ = utils_cf.batch_to_ranking_matrix(
                batch={key: struct_batch}, object_combinations=object_combinations
            )
            assert matrix_test.shape[1] == 1 and len(non_neg_indices_test) == 2
            new_model = cf_model.add_user_to_model(
                model=model,
                new_ratings=matrix_test,
                non_neg_indices=non_neg_indices_test,
                learning_rate=hyperparams["learning_rate"],
                convergence_threshold=hyperparams["convergence_threshold"],
            )
            new_rank_matrix = new_model.forward().detach().cpu().numpy()
            ranker = utils_cf.CFRanker(
                combinations=object_combinations,
                ranking_matrix=new_rank_matrix,
                user_list=user_list_train + [struct_batch["user_id"]],
                object_ID_list_semantic=object_ID_list_semantic,
                semantic_similarity_matrix=semantic_similarity_matrix,
            )
        else:
            ranker = ranker_original
        partial_scene = struct_batch["partial"]
        object_ids = []
        is_unplaced = []
        for surf in partial_scene:
            object_ids.extend(list(o.object_id for o in surf.objects_on_surface))
            if surf.surface_type == constants.SURFACE_TYPES.NOT_PLACED:
                is_unplaced.extend(
                    list(True for _ in surf.objects_on_surface)
                )
            else:
                is_unplaced.extend(
                    list(False for _ in surf.objects_on_surface)
                )

        cluster_labels = utils_cf.cluster_objects(
            ranker=ranker,
            objects=object_ids,
            user_id=struct_batch["user_id"],
            max_clusters=len(partial_scene) - 1,
        )
        surface_name_assignments, cost_removing = utils_cf.object_clusters_to_placement(
            objects=object_ids,
            object_cluster_labels=cluster_labels,
            initial_scene=partial_scene,
        )

        # Generate a new scene while only placing the unplaced objects.
        predicted_scene = deepcopy(partial_scene)
        surface_names = [surf.name for surf in predicted_scene]
        for object_id, assigned_surf, flag in zip(object_ids, surface_name_assignments, is_unplaced):
            if not flag:
                continue
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

        # Calculate evaluation metrics.
        goal_scene = struct_batch["goal"]
        yield {
            "scene_id": key,
            "ranker": ranker,
            "partial_scene": partial_scene,
            "predicted_scene": predicted_scene,
            "goal_scene": goal_scene,
            "cost_removing": cost_removing,
        }


def evaluate(
    hyperparams: Dict[str, Union[int, float]],
    test_batches: Dict[str, Dict[str, Any]],
    checkpoint_folder: Path,
    object_ID_list_semantic: List[str],
    semantic_sim_mat: np.ndarray,
    object_id_dict: Dict[str, int],
):
    """Loads and evaluates CF model on the test set."""
    result_array = []
    prediction_array = {}
    latent_embeddings = {}

    with open(checkpoint_folder / "train_params.pkl", "rb") as fp:
        train_params = pkl.load(fp)
    model = cf_model.CFModel(
        hidden_dimension=hyperparams["hidden_dimension"],
        num_pairs=len(train_params["object_combinations"]),
        num_users=len(train_params["user_list_train"]),
        lambda_reg=hyperparams["lambda_reg"],
        mode="test",
    )
    model.load_state_dict(torch.load(checkpoint_folder / "model.ckpt"))

    response_generator = run_batch(
        model=model,
        hyperparams=hyperparams,
        eval_batches=test_batches,
        user_list_train=train_params["user_list_train"],
        object_combinations=train_params["object_combinations"],
        object_ID_list_semantic=object_ID_list_semantic,
        semantic_similarity_matrix=semantic_sim_mat,
        object_id_dict=object_id_dict,
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
                "container_type": test_batches[key]["container_type"],
                "household": test_batches[key]["household"],
                "num_demonstrations": test_batches[key]["num_demonstrations"],
                "num_removed_demonstrations": sum(
                    int(x) for x in test_batches[key]["num_removed_demonstrations"]
                ),
                "num_misplaced_objects": len(unplaced_objects),
                "num_misplaced_object_classes": len(unplaced_object_classes),
                "edit_distance": edit_distance,
                "igo": igo,
                "ipc": ipc,
                "cost_removing": response["cost_removing"],
            }
        )
        prediction_array[key] = {
            "predicted_scene": utils_data.scene_to_json(response["predicted_scene"]),
            "goal_scene": utils_data.scene_to_json(response["goal_scene"]),
        }
        latent_embeddings[key] = {
            "ranking_matrix": response["ranker"].ranking_matrix,
            "user_list": response["ranker"].user_list,
            "object_combinations": response["ranker"].combinations,
        }  # TODO: remove if redundant.
    return result_array, prediction_array, latent_embeddings
