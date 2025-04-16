"""Helper functions for evaluating rearrangement methods."""

from typing import List, Dict, Tuple, Union
import csv
from pathlib import Path
from copy import deepcopy
import numpy as np
import scipy.optimize as spo

from utils import data_struct
from utils import utils_data


def return_early_stopping_ckpt(
    folder: Path, stopping_metric: str, is_min: bool=True
) -> Path:
    """Choose the checkpoint based on early stopping using a given metric.
    
    Note: Checkpoints must be saved in this format:
        <stopping_metric>=<value>-epoch=<epoch>-step=<step>.ckpt.

    Args:
        folder: Path to the folder containing the results.
        stopping_metric: Metric to use for early stopping.
        is_min: Whether to choose the minimum or maximum value for the metric.
    Returns:
        Path to the best checkpoint and the value of the stopping metric.
    Raises: 
        FileNotFoundError if no checkpoint files are found in the folder.
        ValueError if no checkpoint files are found with the stopping metric.
    """
    checkpoints = list(folder.glob(f"{stopping_metric}=*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {folder.name} with {stopping_metric} early stopping.")
    # TODO: Better way to sort.
    checkpoints = sorted(
        checkpoints, key=lambda x: x.name.split("=", maxsplit=1)[1]
    )

    # Find the best checkpoint based on the stopping metric.
    best_ckpt = None
    best_value = None
    for ckpt in checkpoints:
        metric_value = float(ckpt.name.split("-", maxsplit=1)[0].split("=")[1])
        if (
            best_value is None
            or (is_min and metric_value < best_value)
            or (not is_min and metric_value > best_value)
        ):
            best_ckpt = ckpt.name
            best_value = metric_value
    return best_ckpt, best_value


def load_results(folder):
    """Load results from csv file as an array."""
    if not Path(folder).is_dir():
        raise FileExistsError(f"{folder} is not a valid directory path.")
    with open(Path(folder) / "results.csv", "r") as f:
        reader = csv.DictReader(f)
        results = list(row for row in reader)
    for i, row in enumerate(results):
        for k in row.keys():
            if k == "scene_id" or k == "container_type" or k == "scene_num":
                continue
            elif k == "percent_removed_demos":
                results[i][k] = float(row[k])
            else:
                results[i][k] = int(row[k])
    return results


def _scene_to_cluster(scene):
    """Returns a nested list of objects matching the clustering in the given scene.
    """
    object_clusters = {}
    for clid, surface in enumerate(scene[:-1]):
        object_clusters[f"cluster-{clid}"] = []
        for obj in surface.objects_on_surface:
            object_clusters[f"cluster-{clid}"].append(obj.name)
    return list(object_clusters.values())


def calculate_ipc(predicted_scene, goal_scene):
    """Calculates the number of incorrectly placed object classes.
    
    This function wraps around the edit distance function to calculate the
    number of object classes placed at the wrong location.
    """

    _, unmatched_objects = calculate_edit_distance(predicted_scene, goal_scene)
    unmatched_object_classes = set(obj[0] for obj in unmatched_objects)
    return len(unmatched_object_classes)


def calculate_igo(predicted_scene, goal_scene, verbose=False):
    """Calculates the number of incorrectly grouped objects.
    
    This function compares object clusters in the predicted scene and goal scene
    using linear sum assignment by comparing two clusters using the scene edit
    distance metric. The number of objects deviating from the clustering in
    the goal scene is returned.
    """

    goal_object_clusters = _scene_to_cluster(goal_scene)
    predicted_object_clusters = _scene_to_cluster(predicted_scene)

    # Compare the predicted and goal object clusters by computing the linear sum
    # assignment using jaccard index scores.
    distance_metric = np.zeros(
        (len(goal_object_clusters), len(predicted_object_clusters))
    )
    for i, gc in enumerate(goal_object_clusters):
        for j, pc in enumerate(predicted_object_clusters):
            # Metric: Number of objects in gc that are missing from pc.
            distance_metric[i, j] = len(utils_data.return_unmatched_elements(gc, pc))
    row_array, col_array = spo.linear_sum_assignment(distance_metric, maximize=False)
    igo = distance_metric[row_array, col_array].sum()
    if verbose:
        for row_id, col_id in zip(row_array, col_array):
            print(f"[{distance_metric[row_id, col_id]}] {goal_object_clusters[row_id]} -> {predicted_object_clusters[col_id]}")
    return int(igo)


def object_placements_to_scene_struct(
    object_placements: List[Tuple[data_struct.ObjectEntity, str]],
    surface_names: List[str],
    surface_constants: Dict[str, Dict[str, str]]
) -> List[data_struct.SurfaceEntity]:
    """Converts object placements to a scene struct."""
    scene_struct = []
    for surface in surface_names:
        surface_params = surface_constants[surface]
        scene_struct.append(
            utils_data.generate_surface_instance(
                name=surface, environment=surface_params["environment"],
                surface_type=surface_params["surface_type"],
                position=surface_params["position"], text_label=surface_params["text"],
                objects_on_surface=[]
            )
        )
    scene_struct.append(utils_data.return_unplaced_surface())
    for obj, surface in object_placements:
        if surface == "unplaced_surface":
            raise ValueError("Object cannot be placed on unplaced surface.")
        matching_surface = [
            i for i, surf in enumerate(scene_struct) if surf.name == surface
        ]
        if not matching_surface:
            raise ValueError(f"Surface {surface} not found in scene struct.")
        scene_struct[matching_surface[0]].add_objects([obj])
    return scene_struct


def place_objects_in_scene(
    scene_struct: List[data_struct.SurfaceEntity],
    object_id_surface_tuples: List[Tuple[str, str]]
) -> List[data_struct.SurfaceEntity]:
    """Moves unplaced objects in the given scene to their assigned surfaces
    
    Args:
        scene_struct: Scene struct of the initial environment state.
        object_id_surface_tuples: List of object ID-surface name tuples for
            object IDs lying on the unplaced surface.
    
    Raises: 
        ValueError if any of the object ids in object_id_surface_tuples do not
        match any of the unplaced objects.

    Returns:
        Scene struct with unplaced objects moved to their assigned surfaces.
    """

    updated_scene_struct = list(deepcopy(surf) for surf in scene_struct)
    assert updated_scene_struct[-1].name == "unplaced_surface"

    for object_id, surface in object_id_surface_tuples:
        matching_unplaced_objects = [
            obj for obj in updated_scene_struct[-1].objects_on_surface
            if obj.object_id == object_id
        ]
        if not matching_unplaced_objects:
            raise ValueError(f"Object {object_id} not found in unplaced objects"
                             " but is assigned a surface.")
        matching_surface_index = [
            i for i, surf in enumerate(updated_scene_struct)
            if surf.name == surface
        ]
        # Remove object from unplaced surface and place it on the assigned one.
        updated_scene_struct[-1].objects_on_surface.remove(matching_unplaced_objects[0])
        updated_scene_struct[matching_surface_index[0]].add_objects([
            matching_unplaced_objects[0]
        ])

    return updated_scene_struct


def calculate_edit_distance(
    scene_query: List[data_struct.SurfaceEntity],
    scene_ref: List[data_struct.SurfaceEntity],
) -> Tuple[int, List[Tuple[str, str]]]:
    """Calculates the edit distance between two rearrangement scenes.

    This function finds the number of objects misplaced in scene_query with
    respect to scene_ref by assuming that the surfaces in each scene is unique.
    An exception is raised if the scenes are not identical. (This can
    be extended to find the least number of object movements to get from
    scene_query to scene_ref.)

    Args:
        scene_query: List of surfaces, representing a scene.
        scene_ref: List of surfaces, representing a scene.

    Returns:
        Edit distance between input scenes and a list of misplaced objects with
        each objects position in scene_query.

    Raises:
        ValueError if environment layouts are not identical or scenes have
        different objects.
    """

    # Check if the environment layouts are identical.
    if (
        not all(surface_x in scene_ref for surface_x in scene_query)
        or not all(surface_y in scene_query for surface_y in scene_ref)
        or len(scene_query) != len(scene_ref)
    ):
        raise ValueError("Environment layouts are not identical")

    objects_scene_query = []
    objects_scene_ref = []
    for surface_x, surface_y in zip(scene_query, scene_ref):
        objects_scene_query.extend(surface_x.objects_on_surface)
        objects_scene_ref.extend(surface_y.objects_on_surface)

    sort_by_id = lambda x: x.object_id
    if sorted(objects_scene_query, key=sort_by_id) != sorted(
        objects_scene_ref, key=sort_by_id
    ):
        raise ValueError("Scenes have different sets of objects")

    edit_distance = 0
    unmatched_objects_query = []
    for surface_x in scene_query:
        # Skip the surfaces containing unplaced objects.
        # if surface_x.surface_type == constants.SURFACE_TYPES.NOT_PLACED:
        #     continue
        for surface_y in scene_ref:
            # if surface_y.surface_type == constants.SURFACE_TYPES.NOT_PLACED:
            #     continue
            if surface_x == surface_y:
                unmatched_objects = utils_data.return_unmatched_elements(
                    surface_x.objects_on_surface, surface_y.objects_on_surface
                )
                unmatched_objects_query.extend(
                    [(obj.object_id, surface_x.name) for obj in unmatched_objects]
                )
                edit_distance += len(unmatched_objects)

    return edit_distance, unmatched_objects_query


def compute_eval_metrics(
    scene_query: List[data_struct.SurfaceEntity],
    scene_ref: List[data_struct.SurfaceEntity]
) -> Tuple[int, int, int]:
    """Computes all evaluation metrics between two scenes."""
    edit_distance, _ = calculate_edit_distance(scene_query, scene_ref)
    ipc = calculate_ipc(scene_query, scene_ref)
    igo = calculate_igo(scene_query, scene_ref)
    return edit_distance, ipc, igo
