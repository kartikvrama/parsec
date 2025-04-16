"""Functions for generating ranking matrix for CF."""
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import combinations
import random
import numpy as np
import torch
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from utils import constants
from utils import data_struct
from utils import utils_data

EPS = 1e-6

def batch_to_ranking_matrix(
    batch: Dict[str, Any],
    object_combinations: Optional[List[Tuple[str, str]]]=None
) -> Tuple[np.ndarray, List[str], List[Tuple[str, str]]]:
    """Converts a pkl batch into a ranking matrix for training/evaluating CF."""
    arrangements, user_labels, _ = return_arrangements_from_batch(batch)
    print(f"Number of arrangements in the batch: {len(arrangements)}")
    return generate_ranking_matrix(
        user_labels=user_labels,
        arrangement_list=arrangements,
        object_combinations=object_combinations
    )


def return_arrangements_from_batch(batch):
    """Recovers the list of observed arrangements and metadata from pkl batch."""
    data_tuples = []
    for scene_dict in batch.values():
        user_id = scene_dict["user_id"]
        for label, obs in zip(
            scene_dict["demonstration_labels"],
            scene_dict["observed_arrangement_list"]
        ):
            if any(
                t[1] == user_id and t[2] == label
                for t in data_tuples
            ):  # Skip duplicate entries.
                continue
            data_tuples.append((obs, user_id, label))
    return list(zip(*data_tuples))


class CFRanker:
    """
    Computes the pairwise ranking of objects for known and seen objects.
    # TODO: Complete attributes and descriptions.
    """

    rho_min: float
    combinations: List[Tuple[str, str]]
    user_list: List[str]
    ranking_matrix: np.ndarray
    object_ID_list_semantic: Optional[List[str]]
    semantic_similarity_matrix: Optional[np.ndarray]

    def __init__(
        self,
        combinations: List[Tuple[str, str]],
        ranking_matrix: Union[np.ndarray, torch.Tensor],
        user_list: List[str],
        object_ID_list_semantic: Optional[List[str]] = None,
        semantic_similarity_matrix: Optional[np.ndarray] = None,
        rho_min: float = 0.4,
    ):
        """Initializes the CF ranker.
        
        Args:
            combinations: List of all object combinations with learned rankings.
            ranking_matrix: Matrix of learned object rankings.
            user_list: Ordered list of known users to find the column id.
            object_ID_list_semantic: List of all object IDs with semantic similarity scores.
            semantic_similarity_matrix: Pairwise semantic similarity matrix, 
                with ordering as per object_ID_list_semantic.
            rho_min: Minimum threshold for pairwise distance. Defaults to 0.4.
        """
        self.rho_min = rho_min
        self.combinations = combinations
        self.user_list = user_list
        self.ranking_matrix = ranking_matrix
        if isinstance(self.ranking_matrix, torch.Tensor):
            self.ranking_matrix = self.ranking_matrix.numpy()

        self.object_ID_list_semantic = None
        self.semantic_similarity_matrix = None
        if object_ID_list_semantic is not None and semantic_similarity_matrix is not None:
            if not np.all(
                semantic_similarity_matrix - semantic_similarity_matrix.T < 1e-6
            ):
                raise ValueError("Pairwise distance matrix is not symmetric")
            self.object_ID_list_semantic = object_ID_list_semantic
            self.semantic_similarity_matrix = semantic_similarity_matrix
            self.known_obj_list = list(
                o for o in object_ID_list_semantic if any(o in comb for comb in combinations)
            )
            self.known_rows_sem = list(object_ID_list_semantic.index(o) for o in self.known_obj_list)

        self.unknown_obj_distances = {}  # Cache for unknown object distances.

    def __call__(
        self,
        obj_1: str,
        obj_2: str,
        user_id: str,
    ):
        """Returns the ranking of the object combination for the user.
        
        Args:
            obj_1: Object ID of the first object in the combination.
            obj_2: Object ID of the second object in the combination.
            user_id: User ID to find the column ID.
        """
        if not user_id in self.user_list:
            raise ValueError(f"User ID {user_id} is not in the list of known users.")
        try:
            return self._find_ranking_known_objects(
                obj_1=obj_1,
                obj_2=obj_2,
                user_id=user_id
            )
        except KeyError as e:
            if self.semantic_similarity_matrix is None:
                raise e
            if obj_1 not in self.known_obj_list and obj_2 not in self.known_obj_list:
                # # TODO: check paper, probably return semantic similarity score.
                # raise NotImplementedError(f"Cannot handle calculating similarity between 2 unknown objects: {obj_1} and {obj_2}") from e
                return self.semantic_similarity_matrix[
                    self.object_ID_list_semantic.index(obj_1),
                    self.object_ID_list_semantic.index(obj_2),
                ]
            known_object = obj_1 if obj_1 in self.known_obj_list else obj_2
            unknown_object = obj_2 if known_object == obj_1 else obj_1
            return self._find_ranking_new_objects(
                known_obj=known_object,
                unknown_obj=unknown_object,
                user_id=user_id
            )

    def _find_ranking_known_objects(
        self,
        obj_1: str,
        obj_2: str,
        user_id: str,
    ):
        column_id = self.user_list.index(user_id)
        row_id = objcomb2row(obj_1, obj_2, self.combinations)
        return self.ranking_matrix[row_id, column_id]

    def _find_ranking_new_objects(
        self,
        known_obj: str,
        unknown_obj: str,
        user_id: str,
    ):
        """Calculates the ranking of the unknown object based on the known object.
        
        Args:
            known_obj: Object ID of the known object.
            unknown_obj: Object ID of the unknown object.
            user_id: User ID to find the column ID.
        """
        if (known_obj, unknown_obj, user_id) in self.unknown_obj_distances:
            return self.unknown_obj_distances[(known_obj, unknown_obj, user_id)]

        if known_obj not in self.object_ID_list_semantic or unknown_obj not in self.object_ID_list_semantic:
            raise ValueError(f"Semantic similarity between {known_obj} and {unknown_obj} is unknown")
        term_1 = self.semantic_similarity_matrix[
            self.object_ID_list_semantic.index(known_obj),
            self.object_ID_list_semantic.index(unknown_obj),
        ]

        known_obj_rank_arr = np.array([
            self._find_ranking_known_objects(known_obj, obj_j, user_id) for obj_j in self.known_obj_list
        ])  # r(o_k, o_l)
        known_obj_sem_sim_arr = self.semantic_similarity_matrix[
            self.object_ID_list_semantic.index(known_obj), self.known_rows_sem
        ]  # rho(o_k, o_l)
        unknown_obj_sem_sim_arr = self.semantic_similarity_matrix[
            self.object_ID_list_semantic.index(unknown_obj), self.known_rows_sem
        ]  # rho(o*, o_l)
        if np.all(unknown_obj_sem_sim_arr < self.rho_min):  # If all pairwise distances are below threshold.
            return 0

        # sum_{l}(r(o_*, o_l) * [(r(o_k, o_l) - rho(o_k, o_l)])/sum(rho(o*, o_l))
        term_2 = np.sum(
            unknown_obj_sem_sim_arr*(known_obj_rank_arr - known_obj_sem_sim_arr)
        )/(1e-6 + np.sum(unknown_obj_sem_sim_arr))
        # Add to cache.
        self.unknown_obj_distances[(known_obj, unknown_obj, user_id)] = term_1 + term_2
        return term_1 + term_2


def calculate_mean_error_expert():
    # TODO: Implement this function.
    pass


def objcomb2row(
    obj1: List[str],
    obj2: List[str],
    combinations: List[Tuple[str, str]],
) -> int:
    """Returns the row index of the object combination in the ranking matrix.

    Args:
        obj1: Object ID of first object in the combination.
        obj2: Object ID of second object in the combination.
        combinations: Ordered list of all object combinations.
    Returns:
        Row index of the object combination in the matrix.
    Raises:
        KeyError: If the object combination does not exist.
    """

    for i, objtup in enumerate(combinations):
        if objtup in [(obj1, obj2), (obj2, obj1)]:
            return i
    raise KeyError(f"{obj1}/{obj2} combination does not exist in list")


def return_object_combinations(
    arrangement_list: List[List[data_struct.SurfaceEntity]],
):
    """Returns all nc2 pairs of objects from a list of arrangements."""
    object_id_list = set()
    for agmt in arrangement_list:
        for surf in agmt:
            object_id_list.update([obj.object_id for obj in surf.objects_on_surface])
    return list(combinations(object_id_list, 2)) + list((o, o) for o in object_id_list)


def increment_ranking_matrix_positive(
    matrix,
    obj_group_1: List[str],
    column_id: int,
    combinations: List[Tuple[str, str]],
    ignore_new_objects: bool=False,
):
    """Adds objects within a scene cluster to the (positive) ranking matrix.
    
    Args:
        matrix: Ranking matrix to update.
        obj_group_1: Group of objects in the scene.
        column_id: Column ID to update. This depends on the schema corresponding
            to the scene.
        combinations: List of all object combinations in the matrix.
        ignore_new_objects: If True, ignore objects not in the combination list.
            Raises a KeyError otherwise.
    """
    if not len(combinations) == matrix.shape[0]:
        raise ValueError(f"Combinations list {len(combinations)} does not match the number of matrix rows {matrix.shape[0]}")
    assert obj_group_1 is not None
    for index, obj_1 in enumerate(obj_group_1):
        for obj_2 in obj_group_1[index + 1 :]:
            try:
                row_id = objcomb2row(obj_1, obj_2, combinations)
            except KeyError as e:
                if ignore_new_objects:
                    continue
                raise e
            matrix[row_id, column_id] = 1

def increment_ranking_matrix_negative(
    matrix,
    obj_group_1: List[str],
    obj_group_2: List[str],
    column_id: int,
    combinations: List[Tuple[str, str]],
    ignore_new_objects: bool=False,
):
    """Adds objects across scene clusters to the (negative) ranking matrix.

    This script updates either a positive and negative ranking matrix, depending
    on input obj_group_2. If obj_group_2 is None, all object combinations from
    this set are positive pairs, else Object combinations across object sets
    are negative pairs.

    Args:
        matrix: Ranking matrix to update.
        obj_group_1: First group of objects in the scene.
        obj_group_2: Second group of objects in the scene.
        column_id: Column ID to update. This depends on the schema corresponding
            to the scene.
        combinations: List of all object combinations in the matrix.
    """
    if not len(combinations) == matrix.shape[0]:
        raise ValueError(f"Combinations list {len(combinations)} does not match the number of matrix rows {matrix.shape[0]}")
    assert obj_group_1 is not None and obj_group_2 is not None
    for obj_1 in obj_group_1:
        for obj_2 in obj_group_2:
            try:
                row_id = objcomb2row(obj_1, obj_2, combinations)
            except KeyError as e:
                if ignore_new_objects:
                    continue
                raise e
            matrix[row_id, column_id] = 1


def generate_ranking_matrix(
    user_labels: List[str],
    arrangement_list: List[List[data_struct.SurfaceEntity]],
    object_combinations: Optional[List[Tuple[str, str]]]=None
):
    """Generate a pairwise ranking matrix from the json data.

    The pairwise ranking matrix is generated by combining separate count
    matrices for positive and negative object pairs.

    Args:
        arrangements_labeled: List of object arrangements labeled by the user ID.
        object_combinations: List of all object combinations.
    """

    # Generate all object combinations from the arrangements.
    return_combinations = False
    if object_combinations is None:
        return_combinations = True
        object_combinations = return_object_combinations(arrangement_list)

    user_list = list(set(user_labels))  # List of unique user IDs.
    pos_matrix = np.zeros((len(object_combinations), len(user_list)))
    neg_matrix = np.zeros((len(object_combinations), len(user_list)))
    for user_id, arrangement in zip(user_labels, arrangement_list):
        column_id = user_list.index(user_id)
        # Positive object pairs.
        for surf in arrangement:
            if surf.name == constants.SURFACE_TYPES.NOT_PLACED:
                continue
            increment_ranking_matrix_positive(
                pos_matrix,
                obj_group_1=[obj.object_id for obj in surf.objects_on_surface],
                column_id=column_id,
                combinations=object_combinations,
                ignore_new_objects=not return_combinations
            )  # Ignore new objects if you are generating matrix for evaluation.
        # Negative object pairs.
        for i, surf_i in enumerate(arrangement):
            for surf_j in arrangement[i + 1 :]:
                increment_ranking_matrix_negative(
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
        1,
        final_ranking_matrix
    )
    final_ranking_matrix = np.where(
        (pos_matrix == 1) & (neg_matrix == 1),
        0.5,
        final_ranking_matrix
    )
    final_ranking_matrix = np.where(
        (pos_matrix == 0) & (neg_matrix == 1),
        0,
        final_ranking_matrix
    )
    non_neg_indices = np.nonzero(final_ranking_matrix >= 0)
    if return_combinations:
        return final_ranking_matrix, user_list, non_neg_indices, object_combinations
    return final_ranking_matrix, user_list, non_neg_indices, None


def cluster_objects(
    ranker: CFRanker,
    objects: List[str],
    user_id: str,
    max_clusters: int = 5,
    seed: int = constants.SEED,
):
    """Determines object grouping by clustering objects based on learned rankings.
    
    Args:
        ranker: CF ranker object. See CFRanker class for more details.
        objects: List of all objects in the scene.
        user_id: User ID to find the column ID.
    """
    adj_matrix = -1*np.ones((len(objects), len(objects)))  # Also the laplacian matrix.
    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects[i:]):
            adj_matrix[i, i + j] = ranker(obj_i, obj_j, user_id)
            adj_matrix[i + j, i] = adj_matrix[i, i + j]  # Symmetric matrix.
    assert np.all(adj_matrix - adj_matrix.T < 1e-6), "Affinity matrix is not symmetric"
    if np.any(np.isnan(adj_matrix)):
        raise ValueError("Affinity matrix contains NaN values, possibly due to unstable CF matrix")

    # Calculate number of clusters as number of zero eigenvalues.
    eigen_values_affinity = np.linalg.eigvalsh(adj_matrix)
    # 1 <= num_clusters (or number of 0 eigen values) <= max_clusters
    num_clusters = max(min(np.sum(np.abs(eigen_values_affinity) < 1e-6), max_clusters), 1)

    # Use exponential function for numerical stability when performing spectral clustering.
    adj_matrix_normalized = (adj_matrix - np.mean(adj_matrix))/(np.std(adj_matrix) + EPS)
    adj_matrix_kernel = np.exp(adj_matrix_normalized)

    # Perform spectral clustering.
    spectral_clustering = SpectralClustering(
        n_clusters=num_clusters,
        random_state=seed,
        affinity="precomputed",
        assign_labels="kmeans",
    ).fit(adj_matrix_kernel)
    return spectral_clustering.labels_


def object_clusters_to_placement(
    objects: List[str],
    object_cluster_labels: List[int],
    initial_scene: List[data_struct.SurfaceEntity],
):
    """Converts object clusters to surface placements based on initial object arrangement.
    
    Surfaces in the initial scene are shuffled before calculating linear sum
    assigments to randomly assign clusters to empty surfaces / surfaces with
    identical similarity scores. This function also returns the number of objects
    that need to be moved from the initial scene to achieve the predicted configuration.

    Args:
        objects: IDs of objects in the scene.
        cluster_labels: List of cluster labels for each object.
        initial_scene: Initial (partially arranged) scene.
    Returns:
        List of assigned surface names per object and the cost of
            transforming the initial scene into the predicted configuration.
    """
    # TODO: verify.
    assert initial_scene[-1].surface_type == constants.SURFACE_TYPES.NOT_PLACED, "Last surface should be NOT_PLACED"
    object_clusters_predicted = []  # List of object clusters arranged by label number.
    for label in sorted(list(set(object_cluster_labels))):
        object_clusters_predicted.append(
            [objects[i] for i, l in enumerate(object_cluster_labels) if l == label]
    )

    if all(not surf.objects_on_surface for surf in initial_scene[:-1]):
        # If no objects are placed, randomly assign surfaces.
        shuffled_surfaces = list(surf.name for surf in initial_scene[:-1])
        random.shuffle(shuffled_surfaces)
        object_label_to_surface_dict = dict({
            i: s for i, s in enumerate(shuffled_surfaces)
        })
        # TODO: calculate cost of moving objects from initial scene.
        return list(object_label_to_surface_dict[l] for l in object_cluster_labels), None

    object_clusters_initial = []
    surface_names = []
    for surf in initial_scene[:-1]:
        object_clusters_initial.append([obj.object_id for obj in surf.objects_on_surface])
        surface_names.append(surf.name)
    indices = np.arange(len(surface_names))
    random.shuffle(indices)  # Shuffle the order of surfaces.
    object_clusters_initial = [object_clusters_initial[i] for i in indices]
    surface_names = [surface_names[i] for i in indices]

    # Hungarian algorithm to match object clusters.
    def _cost_function(cluster_pred, cluster_init):
        # Returns the number of objects that need to be moved from initial cluster.
        items_to_remove = utils_data.return_unmatched_elements(
            cluster_init, cluster_pred
        )  # Objs in initial cluster that have to be removed.
        return len(items_to_remove)
    cost_matrix = np.array([
        [_cost_function(obj_cluster_p, obj_cluster_i) for obj_cluster_i in object_clusters_initial]
        for obj_cluster_p in object_clusters_predicted
    ])  # Similarity matrix between predicted (rows) and intial object clusters (columns).
    predicted_initial_matching = linear_sum_assignment(cost_matrix, maximize=False)
    object_label_to_surface_dict = dict({
        m[0]: surface_names[m[1]] for m in zip(*predicted_initial_matching)
    })
    return (
        list(object_label_to_surface_dict[l] for l in object_cluster_labels),
        sum(list(cost_matrix[i, j] for i, j in zip(*predicted_initial_matching)))
    )
