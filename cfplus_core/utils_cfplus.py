"""Functions for generating ranking matrices for the CF+ baseline."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from utils import constants
from utils import data_struct
from cf_core import utils_cf


def batch_to_ranking_matrix(
    batch: Dict[str, Any],
    object_combinations: Optional[List[Tuple[str, str]]]=None
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Converts a pkl batch into a ranking matrix for training/evaluating CF."""
    arrangements, user_labels, _ = utils_cf.return_arrangements_from_batch(batch)
    return generate_ranking_matrix_cfplus(
        user_labels=user_labels,
        arrangement_list=arrangements,
        object_combinations=object_combinations
    )


def batch_to_observation_dict(
    batch: Dict[str, Any]
) -> Dict[str, List[List[data_struct.SurfaceEntity]]]:
    """Converts a pkl batch into a dictionary of user observations."""
    arrangements, user_labels, _ = utils_cf.return_arrangements_from_batch(batch)
    observation_dict = {}
    for user, arrangement in zip(user_labels, arrangements):
        if user not in observation_dict:
            observation_dict[user] = []
        observation_dict[user].append(arrangement)
    return observation_dict


def generate_ranking_matrix_cfplus(
    user_labels: List[str],
    arrangement_list: List[List[data_struct.SurfaceEntity]],
    object_combinations: Optional[List[Tuple[str, str]]]=None
):
    """Generate a pairwise ranking matrix from the json data.

    This is different from the ranking matrix generator in the CF model. Positive
    rankings are valued at 0, negative rankings are valued at 1, mixed rankings
    and unknown rankings remain the same at 0.5 and -1 respectively.

    Args:
        arrangements_labeled: List of object arrangements labeled by the user ID.
        object_combinations: List of all object combinations.
    """

    # Generate all object combinations from the arrangements.
    return_combinations = False
    if object_combinations is None:
        return_combinations = True
        object_combinations = utils_cf.return_object_combinations(arrangement_list)

    user_list = list(set(user_labels))  # List of unique user IDs.
    pos_matrix = np.zeros((len(object_combinations), len(user_list)))
    neg_matrix = np.zeros((len(object_combinations), len(user_list)))
    for user_id, arrangement in zip(user_labels, arrangement_list):
        column_id = user_list.index(user_id)
        # Positive object pairs.
        for surf in arrangement:
            if surf.name == constants.SURFACE_TYPES.NOT_PLACED:
                continue
            utils_cf.increment_ranking_matrix_positive(
                pos_matrix,
                obj_group_1=[obj.object_id for obj in surf.objects_on_surface],
                column_id=column_id,
                combinations=object_combinations,
                ignore_new_objects=not return_combinations
            )  # Ignore new objects if you are generating matrix for evaluation.
        # Negative object pairs.
        for i, surf_i in enumerate(arrangement):
            for surf_j in arrangement[i + 1 :]:
                utils_cf.increment_ranking_matrix_negative(
                    neg_matrix,
                    obj_group_1=[obj.object_id for obj in surf_i.objects_on_surface],
                    obj_group_2=[obj.object_id for obj in surf_j.objects_on_surface],
                    column_id=column_id,
                    combinations=object_combinations,
                    ignore_new_objects=not return_combinations
                )  # Ignore new objects if you are generating matrix for evaluation.

    final_ranking_matrix = -1*np.ones_like(pos_matrix)
    final_ranking_matrix = np.where(
        (pos_matrix == 1) & (neg_matrix == 0),
        0,
        final_ranking_matrix
    )
    final_ranking_matrix = np.where(
        (pos_matrix == 1) & (neg_matrix == 1),
        0.5,
        final_ranking_matrix
    )
    final_ranking_matrix = np.where(
        (pos_matrix == 0) & (neg_matrix == 1),
        1,
        final_ranking_matrix
    )
    print(f"Number of positive pairs: {np.sum(final_ranking_matrix == 0)}")
    print(f"Number of negative pairs: {np.sum(final_ranking_matrix == 1)}")
    print(f"Total number of known pairs: {np.sum(final_ranking_matrix <= 1)}")
    known_indices = np.nonzero(final_ranking_matrix >= 0)  # Ignoring unknown pairs.
    if return_combinations:
        return final_ranking_matrix, user_list, known_indices, object_combinations
    return final_ranking_matrix, user_list, known_indices, None


def generate_vectors_from_observations(
    user_observation_dict: Dict[str, Any]
):
    """Vectorize object placements for running FM."""
    data_list = []
    label_list = []
    surface_metadata_list = None
    num_surfaces = None
    num_surface_types = None
    for user, arrangement_list in user_observation_dict.items():
        for arrangement in arrangement_list:
            if surface_metadata_list is None:
                surface_metadata_list = [
                    {
                        "name": surf.name,
                        "surface_type": surf.surface_type
                    }
                    for surf in arrangement
                    if surf.surface_type != constants.SURFACE_TYPES.NOT_PLACED
                ]
            if num_surfaces is None:
                num_surfaces = len(surface_metadata_list)
            if num_surface_types is None:
                num_surface_types = len(
                    set(d["surface_type"] for d in surface_metadata_list)
                )

            for surf in arrangement:
                if surf.surface_type == constants.SURFACE_TYPES.NOT_PLACED:
                    assert len(surf.objects_on_surface) == 0, "Unplaced surface should not have objects."
                    continue
                for obj in surf.objects_on_surface:
                    data_list.append(
                        {
                            "user_id": user,
                            "object_id": obj.object_id,
                            "surface_name": surf.name,
                            "surface_type": surf.surface_type,
                        }
                    )
                    label_list.append(0)  # Positive examples assigned 0.
                    data_list.extend([
                        {
                            "user_id": user,
                            "object_id": obj.object_id,
                            "surface_name": x["name"],
                            "surface_type": x["surface_type"],
                        }
                        for x in surface_metadata_list
                        if x["name"] != surf.name
                    ])
                    # Negative examples assigned 1.
                    label_list.extend([1 for _ in range(num_surfaces - 1)])
    assert len(data_list) == len(label_list), f"Data list {len(data_list)} and label list {len(label_list)} lengths do not match."
    vectorizer = DictVectorizer(sparse=False)
    data_vector = vectorizer.fit_transform(data_list)
    return data_vector, np.array(label_list), vectorizer, surface_metadata_list
