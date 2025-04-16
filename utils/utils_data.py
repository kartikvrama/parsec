"""Helper functions for loading and visualizing semantic rearrangement data."""
from typing import Dict, Optional, Tuple, List, Any, Union
from pathlib import Path
import csv
import json
from copy import deepcopy
import pandas as pd
from termcolor import colored

from utils import constants
from utils import data_struct

# Constants
PARENT_DIR = Path(__file__).resolve().parents[1]
OBJ_LABEL_FILE = PARENT_DIR / "labels/object_labels_2024_06_26.csv"
SURF_LABEL_FILE = PARENT_DIR / "labels/surface_labels_2024_02_29.json"
USER_LOGS = PARENT_DIR / "labels/user-log-2024-05-30.csv"


def visualize_scene_with_errors(
    scene: List[data_struct.SurfaceEntity], wrong_placements: List[Tuple[str, str]] 
):
    """Prints a textual representation of the scene and marks incorrect placements.
    """

    print(f"----\nSurface names: {', '.join([surface.name for surface in scene])}")
    print("----Scene----")
    for surface in scene:
        string = f"{surface.text_label:<38}: "
        if not surface.objects_on_surface:
            string += "<EMPTY>  "
        for obj in surface.objects_on_surface:
            if (obj.object_id, surface.name) in wrong_placements:
                string += colored(f"{obj.name}", "red") + ", "
            else:
                string += f"{obj.name}, "
        string += "\b\b"
        print(f"{string}")
    print("------------------")


def return_filtered_fold(
    df: pd.DataFrame,
    num_demonstations: int,
    environment_cat: Optional[str]=None,
    environment_var: Optional[int]=None,
) -> pd.DataFrame:
    """Filters the fold dataframe based on the given arguments."""
    if df.empty:
        raise ValueError("Empty dataframe cannot be filtered.")
    if environment_cat is None or environment_var is None:
        if environment_cat is None and environment_var is None:
            df_filtered = df[df["num_demonstrations"].eq(num_demonstations)]
        else:
            raise ValueError("Neither or both environment category and variant must be provided.")
    else:
        df_filtered = df[
            df["num_demonstrations"].eq(num_demonstations)
            & df["environment"].eq(environment_cat)
            & df["variant"].eq(int(environment_var))
        ]
    if df_filtered.empty:
        return None
    return df_filtered


def return_fold_max_observations(
    df: pd.DataFrame,
    user_list: List[str],
    environment_cat: Optional[str]=None,
    environment_var: Optional[int]=None,
    verbose: bool=False
):
    """Returns examples with the maximum possible observations per user."""
    acq_users = find_eligible_users()
    eligible_users = list(set(user_list) & set(acq_users))
    observations_per_user_dict = {
        uid: df[df["user_id"].eq(uid)]["num_demonstrations"].max()
        for uid in eligible_users
    }
    if verbose:
        for uid, num in observations_per_user_dict.items():
            print(f"User {uid} has maximum of {num} observed arrangements.")
    num_observations_per_user = pd.DataFrame.from_dict(
        {i:(user, num) for i, (user, num) in enumerate(observations_per_user_dict.items()) if num > 0},
        orient="index", columns=["user_id", "num_observations"]
    )
    df_accumulate = pd.DataFrame()
    num_obs_unique = num_observations_per_user["num_observations"].unique().tolist()
    if verbose:
        print(f"Unique number of arrangements: {num_obs_unique}")
    for num_obs in num_obs_unique:
        users_with_num_obs = num_observations_per_user[
            num_observations_per_user["num_observations"].eq(num_obs)
        ]["user_id"].tolist()
        if verbose:
            print(f"Number of users with {num_obs} arrangements: {len(users_with_num_obs)}")
        df_filtered = return_filtered_fold(
            df, num_obs, environment_cat, environment_var
        )
        if df_filtered is None:
            raise ValueError(f"No examples found for {num_obs} arrangements.")
        df_filtered = df_filtered[df_filtered["user_id"].isin(users_with_num_obs)]
        if df_filtered.empty:
            raise ValueError(f"No examples found for {set(users_with_num_obs)} users.")
        df_accumulate = pd.concat([df_accumulate, df_filtered], ignore_index=True)
    if df_accumulate.empty:
        raise ValueError("Failed to extract user examples with max observation length, returned empty data frame.")

    # Raise warning if all user examples with the maximum number of observations per user are included.
    num_empty_scene_examples = len(df_accumulate[df_accumulate["num_remaining_partial"].eq(0)])
    expected_num_empty_scene_examples = sum(1 + num_observations_per_user["num_observations"])
    if expected_num_empty_scene_examples != num_empty_scene_examples:
        print(
            f"WARNING: Number of empty scene examples {num_empty_scene_examples} does not match the sum of arrangements {expected_num_empty_scene_examples}."
        )
    return df_accumulate


def determine_variant(env_variant_labels: Dict[str, str]) -> str:
    """Determines the container variant from a dictionary of variant labels."""

    if (
        env_variant_labels["KitchenCabinetVariant"] == "KitchenCabinet_3X2"
        and env_variant_labels["BathroomCabinetVariant"] == "BathroomCabinet_2hs_1os"
        and env_variant_labels["FridgeVariant"] == "Fridge_4cs_0dr_3ds"
        and env_variant_labels["BookshelfVariant"] == "Bookshelf_5s"
        and env_variant_labels["DresserDrawerVariant"] == "DresserDrawer_6dr_3s_1ts"
    ):
        return 1
    elif (
        env_variant_labels["KitchenCabinetVariant"] == "KitchenCabinet_3X1"
        and env_variant_labels["BathroomCabinetVariant"] == "BathroomCabinet_0hs_5os"
        and env_variant_labels["FridgeVariant"] == "Fridge_3cs_3dr_0ds"
        and env_variant_labels["BookshelfVariant"] == "Bookshelf_4s"
        and env_variant_labels["DresserDrawerVariant"] == "DresserDrawer_5dr_0s_1ts"
    ):
        return 2
    elif (
        env_variant_labels["KitchenCabinetVariant"] == "KitchenCabinet_2X2"
        and env_variant_labels["BathroomCabinetVariant"] == "BathroomCabinet_4hs_0os"
        and env_variant_labels["FridgeVariant"] == "Fridge_2cs_1dr_3ds"
        and env_variant_labels["BookshelfVariant"] == "Bookshelf_6s"
        and env_variant_labels["DresserDrawerVariant"] == "DresserDrawer_2dr_4s_1ts"
    ):
        return 3
    else:
        raise ValueError(
            "Invalid environment set, please check the variant labels."
        )


def find_eligible_users():
    """Returns a list of user IDs that passed the ACQs."""
    with open(USER_LOGS, "r") as fcsv:
        csv_data = csv.DictReader(fcsv)
        eligible_users = list(
            row["User ID"] for row in csv_data if row["ACQ"].startswith("PASS")
        )
    print(f"Number of users passing ACQ: {len(eligible_users)}")
    return eligible_users


def _min_objects(scene_struct: List[data_struct.SurfaceEntity]):
    """Returns True if there are atleast 2 object types arranged."""
    objects = []
    for surf in scene_struct:
        objects.extend([o.object_id for o in surf.objects_on_surface])
    return len(set(objects)) >= 2


def _is_different(
    scene_struct_a: List[data_struct.SurfaceEntity], scene_struct_b: List[data_struct.SurfaceEntity]
):
    """Returns True if the arrangements are different."""
    if not all(s_a == s_b for s_a, s_b in zip(scene_struct_a, scene_struct_b)):
        raise ValueError("Scenes have different surfaces.")
    return not all(
        s_a == s_b and s_a.objects_on_surface == s_b.objects_on_surface
        for s_a, s_b in zip(scene_struct_a, scene_struct_b)
    )


def filter_arrangements(arrangement_list: List[List[data_struct.SurfaceEntity]]):
    """Filters a list of user arrangements based on conditions.
    Refer to the _min_objects and _is_different functions for the conditions.
    """
    filtered_indices = []
    filtered_data = []
    for i, scene in enumerate(arrangement_list):
        if _min_objects(scene):
            if all(_is_different(scene, f) for f in filtered_data):
                filtered_indices.append(i)
                filtered_data.append(scene)
    return filtered_indices, filtered_data


def return_filtered_arrangement_labels(data_folder: Union[str, Path]):
    """Returns a list of filtered arrangement labels for each user."""
    object_id_dict, surface_constants = return_object_surface_constants()
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    json_files = data_folder.rglob("*.json")
    filtered_label_dict = {}
    labels = list("ABCDEF")
    for file in json_files:
        arrangement_list, metadata = load_user_scene_data(
            file, object_id_dict, surface_constants
        )
        filtered_indices, _ = filter_arrangements(arrangement_list)
        filtered_label_dict[metadata["user_id"]] = list(labels[i] for i in filtered_indices)
    return filtered_label_dict


def return_objects_used_per_arrangement(data_folder: Union[str, Path]):
    """Returns a dictionary of objects used per user arrangement."""
    object_id_dict, surface_constants = return_object_surface_constants()
    if isinstance(data_folder, str):
        data_folder = Path(data_folder)
    json_files = data_folder.rglob("*.json")
    arrangement_obj_dict = {}
    labels = list("ABCDEF")
    for file in json_files:
        arrangement_list, metadata = load_user_scene_data(
            file, object_id_dict, surface_constants
        )
        user_id = metadata["user_id"]
        arrangement_obj_dict[user_id] = {i:set() for i in labels}
        for arg_label, scene in zip(labels, arrangement_list):
            for surf in scene:
                arrangement_obj_dict[user_id][arg_label].update(
                    [o.object_id for o in surf.objects_on_surface]
                )
            if not arrangement_obj_dict[user_id][arg_label]:
                print(f"WARNING: Empty arrangement for {user_id} {arg_label}")
    return arrangement_obj_dict


def return_object_surface_constants():
    """Returns dictionaries of object labels and surface constants."""
    object_label_dict = {}
    with open(OBJ_LABEL_FILE, "r") as fcsv:
        csv_data = csv.DictReader(fcsv)
        for row in csv_data:
            object_label_dict[row["fullId"]] = row
    with open(SURF_LABEL_FILE, "r") as fjson:
        surface_constants = json.load(fjson)
    return object_label_dict, surface_constants


def return_unmatched_elements(list_query: List[Any], list_ref: List[Any]) -> List[Any]:
    """Returns items in the query list that do not match with reference list items.

    Args:
        list_query: List of items to be matched.
        list_ref: List of items to be matched against.

    Returns:
        List of unmatched elements in the query list.
    """

    unmatched_elems_query = list_query.copy()
    unmatched_elems_value = list_ref.copy()

    for obj in list_query:
        if obj in unmatched_elems_value:
            # Remove obj from original lists
            unmatched_elems_query.remove(obj)
            unmatched_elems_value.remove(obj)

    return unmatched_elems_query


def scene_to_json(scene: List[data_struct.SurfaceEntity]) -> Dict[str, Any]:
    """Wrapper around SurfaceEntity method to convert a scene to a JSON object.

    Args:
        scene: List of SurfaceEntity instances.

    Returns:
        Nested dictionary of surfaces and objects on each surface.
    """

    return {surface.name: surface.to_dict() for surface in scene}


def json_to_scene(
    json_scene: Dict[str, Any], surface_constants: Dict[str, Dict[str, Any]]
) -> List[data_struct.SurfaceEntity]:
    """Returns a SurfaceEntity from a dictionary version of a scene struct.
    
    Args:
        json_scene: JSON object of a scene. See the to_dict() method for the
            JSON keys.

    Returns:
        List of SurfaceEntity instances.
    """

    scene_struct = []
    for _, surface_dict in json_scene.items():
        surface_params = surface_constants[surface_dict["surface_name"]]
        surface_struct = generate_surface_instance(
            name=surface_dict["surface_name"], environment=surface_params["environment"],
            surface_type=surface_params["surface_type"], position=surface_params["position"],
            text_label=surface_params["text"]
        )
        object_structs = [
            data_struct.ObjectEntity(
                object_id=object_dict["object_id"],
                name=object_dict["object_name"],
            ) for object_dict in surface_dict["objects_on_surface"]
        ]
        surface_struct.add_objects(objects_to_add=object_structs)
        scene_struct.append(surface_struct)

    return scene_struct


def return_intersection_exclusion_indices(list_a: List[Any], list_b: List[Any]):
    """Returns the indices of the bigger list which belong to intersection and
        exclusion. This assumes that elements are not unique.
    """
    if len(list_a) == len(list_b):
        bigger_list = list_a
        smaller_list = list_b
    else:        
        bigger_list = list_a if len(list_a) > len(list_b) else list_b
        smaller_list = list_a if len(list_a) < len(list_b) else list_b

    if len(smaller_list) == 0:
        return [], list(range(len(bigger_list)))
    intersection_indices = []
    exclusion_indices = []

    for big_id, big_item in enumerate(bigger_list):
        for small_id, small_item in enumerate(smaller_list):
            if big_item == small_item:
                intersection_indices.append(big_id)
                del smaller_list[small_id]
                break

    exclusion_indices = [i for i in range(len(bigger_list)) if i not in intersection_indices]
    assert sorted(intersection_indices + exclusion_indices) == list(range(len(bigger_list))), (
        f" intersection: {intersection_indices}, exclusion: {exclusion_indices}"
        f", len of bigger list: {len(bigger_list)}"
    )
    return intersection_indices, exclusion_indices


def find_unplaced_object_placement(
    partial_scene: List[data_struct.SurfaceEntity],
    goal_scene: List[data_struct.SurfaceEntity]
):
    """Matches unplaced objects to their placement in the goal scene."""
    assert all(
        partial_scene[i].name == goal_scene[i].name for i in range(len(partial_scene))
    )
    objectID_surf_pairs = []
    surface_name_list = [surf.name for surf in partial_scene[:-1]]
    for sfid, surf_name in enumerate(surface_name_list):
        unmatched_objects = return_unmatched_elements(
            [obj.object_id for obj in goal_scene[sfid].objects_on_surface],
            [obj.object_id for obj in partial_scene[sfid].objects_on_surface]
        )
        objectID_surf_pairs.extend(
            [(obj_id, surf_name) for obj_id in unmatched_objects]
        )
    return objectID_surf_pairs


def find_all_object_placements(
    partial_scene: List[data_struct.SurfaceEntity], goal_scene: List[data_struct.SurfaceEntity],
    include_empty_surfaces: bool = False
):
    """Match objects to their goal scene placements, split by placed/unplaced objects.
    
    If include_empty_surfaces is True, the function will include empty surfaces in
    the returned pairs."""
    placed_objectID_surf_pairs = []
    unplaced_objectID_surf_pairs = []

    assert goal_scene[-1].surface_type == constants.SURFACE_TYPES.NOT_PLACED
    assert partial_scene[-1].surface_type == constants.SURFACE_TYPES.NOT_PLACED
    goal_surface_list = [surf.name for surf in goal_scene[:-1]]
    partial_surface_list = [surf.name for surf in partial_scene[:-1]]
    for goal_surf_id, surf_name in enumerate(goal_surface_list):
        tmp_goal_list = deepcopy(goal_scene[goal_surf_id].objects_on_surface)
        tmp_partial_list = deepcopy(
            partial_scene[partial_surface_list.index(surf_name)].objects_on_surface
        )
        # Include empty surfaces in the partial scene.
        if not tmp_partial_list and include_empty_surfaces:
            placed_objectID_surf_pairs.append(
                (return_dummy_object(), surf_name)
            )
        for obj in tmp_goal_list:
            if obj in tmp_partial_list:
                placed_objectID_surf_pairs.append((obj, surf_name))
                tmp_partial_list.remove(obj)
            else:
                unplaced_objectID_surf_pairs.append((obj, surf_name))
    return placed_objectID_surf_pairs, unplaced_objectID_surf_pairs


def return_unplaced_surface():
    """Returns an empty unplaced surface"""
    return generate_surface_instance(
        name="unplaced_surface", environment="not_placed",
        surface_type="not_placed", position="None", text_label="None",
        objects_on_surface=None
    )


def clone_surface_type(surface: data_struct.SurfaceEntity) -> data_struct.SurfaceEntity:
    """Clones a surface entity with the same attributes but no objects."""
    position = surface.position if surface.position not in [None, "None"] else "None"
    return generate_surface_instance(
        name=surface.name, environment=surface.environment,
        surface_type=surface.surface_type, position=position,
        text_label=surface.text_label, objects_on_surface=None
    )


def return_dummy_object() -> data_struct.ObjectEntity:
    """Returns a dummy object."""
    return data_struct.ObjectEntity(
        object_id=constants.EMPTY_LABEL, name=constants.EMPTY_LABEL
    )


def remove_object_from_scene(
    scene: List[data_struct.SurfaceEntity], object_id: str, move_to_unplaced: bool
):
    """Remove a specific object from the scenes."""
    scene_copy = [deepcopy(s) for s in scene]
    assert scene[-1].name == "unplaced_surface"
    for sfid, sf in enumerate(scene_copy[:-1]):
        for obj in sf.objects_on_surface:
            if obj.object_id == object_id:
                scene[sfid].objects_on_surface.remove(obj)
                if move_to_unplaced:
                    scene[-1].add_objects([obj])
                return
    raise ValueError(f"Object {object_id} not found in scene.")


def generate_surface_instance(
    name: str, environment: str, surface_type: str, position: Union[List[int], str],
    text_label: str, objects_on_surface: Optional[List[data_struct.ObjectEntity]] = None,
) -> data_struct.SurfaceEntity:
    """Generates a surface object from the name.

    Args:
        name: See SurfaceEntity.name attribute.
        objects_on_surface: Objects placed on this surface. See
            SurfaceEntity.objects_on_surface attribute.

    Returns:
        Instance of SurfaceEntity class.
    """

    assert position == "None" or isinstance(position, List), f"Invalid position type: {position}."
    position=[int(p) for p in position] if isinstance(position, List) else None
    surface_instance = data_struct.SurfaceEntity(
        name=name,
        environment=environment,
        surface_type=surface_type,
        position=position,
        text_label=text_label,
    )
    if objects_on_surface:
        surface_instance.add_objects(objects_to_add=objects_on_surface)
    return surface_instance


def visualize_scene(scene: List[data_struct.SurfaceEntity]):
    """Prints a textual representation of the scene.

    This function DOES NOT sort the objects on each surface in order to
    accurately represent the order in which objects are stacked in the surface
    list.

    Args:
        scene: List of surface entities.
    """

    print(f"----\nSurface names: {', '.join([surface.name for surface in scene])}")
    print("----Scene----")
    for surface in scene:
        string = f"{surface.text_label:<38}: "
        if not surface.objects_on_surface:
            string += "<EMPTY>  "
        for obj in surface.objects_on_surface:
            string += f"{obj.name}, "
        string += "\b\b"
        print(f"{string}")
    print("------------------")


def load_json_arrangement(
    data: Dict[str, Any], object_label_dict: Dict[str, str],
    surface_constants: Dict[str, Dict[str, Any]]
):
    """Loads a single user arranged scene."""
    if not "timeElapsed" in data:
        raise ValueError("'timeElapsed' not found in dataArray.")
    time_taken = data["timeElapsed"]
    scene = []
    for bin_dict in data["finalBins"]:
        bin_name = bin_dict["binName"]
        object_ids = bin_dict["objectIdsInBin"]
        object_instances = [
            data_struct.ObjectEntity(
                object_id=object_id,
                name=object_label_dict[object_id]["text"],
            )
            for object_id in object_ids
        ]
        surface_params = surface_constants[bin_name]
        surface = generate_surface_instance(
            name=bin_name, environment=surface_params["environment"],
            surface_type=surface_params["surface_type"],
            position=surface_params["position"], text_label=surface_params["text"],
            objects_on_surface=object_instances
        )
        scene.append(surface)
    return scene, time_taken


def load_user_scene_data(
    file_path: str, object_label_dict: Dict[str, str],
    surface_constants: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, List[data_struct.SurfaceEntity]], Dict[str, Any]]:
    """Loads object arrangements from a JSON file of user data.

    This function returns each scene as a list of surface entities. User
    metadata such as user ID, problem set index, etc. are also returned.

    Args:
        file_path: Path to the json file.
        object_label_dict: Dictionary mapping object names to their text labels.

    Returns:
        Dictionary of scenes (list of surface entities) and a dictionary
        containing metadata about the user.

    Raises:
        ValueError: If individual and total time taken is not in the JSON file.
    """

    with open(file_path, "r") as fjson:
        json_data = json.load(fjson)
    if not "totalTimeElapsed" in json_data["Final_comments_data"]:
        raise ValueError(
            "'totalTimeElapsed' not found in Final_comments_data section."
        )
    # Store scene metadata for this user.
    user_metadata = {
        "user_id": json_data["turkId"],
        "problem_set": json_data["problemSetIndex"],
        "container_type": constants.PAGE_TO_ENVIRONMENT[json_data["containerType"]],
        "household": determine_variant(json_data["environmentVariants"]),
        "total_time_elapsed": json_data["Final_comments_data"]["totalTimeElapsed"],
    }
    # Construct scene structs from json data of user arrangements.
    arrangement_array = [] # Array of object arrangements.
    time_per_arrangement = [] # Time taken per arrangement.
    for data in json_data["dataArray"]:
        scene, time_taken = load_json_arrangement(
            data, object_label_dict, surface_constants
        )
        arrangement_array.append(scene)
        time_per_arrangement.append(time_taken)
    return arrangement_array, user_metadata


def report_object_statistics(
    scenes_list: List[Dict[str, Any]],
    object_label_file: str = "./labels/object_labels_2024_02_29.csv"
):
    """Reports statistics about the number of objects used.
    
    The function prints out the number of objects used, split by container type
    and variant.

    Args:
        scenes_list: List of user arranged scenes.
        object_label_file: Path to file of object names and their labels.
            Defaults to the csv file in generate_scenes folder.
    """
    # TODO: update.
    raise NotImplementedError
