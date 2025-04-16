"""Generate data splits for different types of experiments."""

from typing import Any, Dict, List
import sys
import json
import random
from pathlib import Path
import pickle as pkl
from absl import app, flags
import pandas as pd

parent_path = Path("../")
sys.path.append(str(parent_path.absolute()))
from utils import constants
from utils import utils_data

flags.DEFINE_string(
    "user_data_dir",
    None,
    "Path to the folder containing user arrangement data for filtering scenes.",
)
flags.DEFINE_string(
    "dataset",
    None,
    "Path to the dataset folder containing permuted scenes.",
)
flags.DEFINE_string(
    "user_list",
    None,
    "Path to the text file containing user IDs to use.",
)
flags.DEFINE_string(
    "destination",
    "myfavoritefolder",
    "Destination folder to save data in.",
)
flags.DEFINE_bool(
    "in_distribution", False, "Generate in-distribution evaluation folds."
)
flags.DEFINE_bool(
    "out_distribution",
    False,
    "Generate out-of-distribution evaluation folds.",
)
flags.DEFINE_bool(
    "unseen_env", False, "Generate unseen environment evaluation folds."
)
flags.DEFINE_bool(
    "unseen_cat", False, "Generate unseen environment category evaluation folds."
)
flags.DEFINE_bool(
    "unseen_obj", False, "Generate unseen object evaluation folds."
)
FLAGS = flags.FLAGS

# Constants
MODES = ["train", "val", "test"]
PD_COLS = [
    "scene_id",
    "user_id",
    "environment",
    "variant",
    "num_removed_goal",
    "num_remaining_partial",
    "num_demonstrations"
]
# Set seed for reproducibility.
random.seed(constants.SEED)


def _filter_glossary(
    scene_metadata: Dict[str, str],
    eligible_users: List[str],
    filtered_arrangement_dict: Dict[str, List[str]],
):
    """Filter scene metadata (glossary) to only eligible users and arrangements.

    Also returns a dictionary of users and their metadata.
    """
    scene_metadata_filtered = {}
    user_metadata = {}
    for key, value in scene_metadata.items():
        if all(int(x) == 0 for x in value["num_removed_demonstrations"]):
            if (
                value["user_id"] in eligible_users
                and value["goal_label"] in filtered_arrangement_dict[value["user_id"]]
                and all(
                    d in filtered_arrangement_dict[value["user_id"]]
                    for d in value["demonstration_labels"]
                )
            ):
                scene_metadata_filtered[key] = value
                if value["user_id"] not in user_metadata:
                    user_metadata[value["user_id"]] = {
                        "container_type": value["container_type"],
                        "household": value["household"],
                    }
    return scene_metadata_filtered, user_metadata


def _append_dict_to_pd(key, value, df):
    """Appends new scene with metadata to a pandas dataframe."""
    new_data = {
        "scene_id": key,
        "user_id": value["user_id"],
        "environment": value["container_type"],
        "variant": int(value["household"]),
        "num_remaining_partial": int(value["num_remaining_partial"]),
        "num_removed_goal": int(value["num_removed_goal"]),
        "num_demonstrations": int(value["num_demonstrations"]),
    }
    df = pd.concat([df, pd.DataFrame(new_data, index=[0])], ignore_index=True)
    return df


def _print_fold_statistics(
    fold_list: List[Dict[str, List[str]]], metadata: Dict[str, Any] = None
):
    """Prints number of examples in each data split of each fold."""
    if metadata is not None:
        for key, value in metadata.items():
            print(f"{key}")
            if isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, list):
                        print(f"Fold {i+1}: {len(v)} users")
                    elif isinstance(v, str):
                        print(f"Fold {i+1}: {v}")
                    else:
                        print("Unknown type.")
            elif isinstance(value, dict):
                for k, v in value.items():
                    print(f"User {k}: {len(v)} folds")
            print("----")

    for i, fold in enumerate(fold_list):
        print(f"Fold {i+1}")
        for split_label, split in fold.items():
            print(f"Split: {split_label}")
            for num_demo in range(6):
                split_df_demo = split[split["num_demonstrations"].eq(num_demo)]
                print(f"Number of examples with {num_demo} demonstrations: {len(split_df_demo)}")
            print(f"TOTAL Number of examples: {len(split)}")
            print("----")


def _save_fold(
    fold_list: List[Dict[str, List[str]]], metadata: Dict[str, Any], destination: Path
):
    """Save the fold data to a pickle file."""
    with open(destination, "wb") as f:
        data = {f"fold_{i+1}": l for i, l in enumerate(fold_list)}
        data.update({"metadata": metadata})
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def fold_data_in_distribution(
    scene_metadata: Dict[str, str],
    user_list: List[str],
    filtered_arrangement_dict: Dict[str, List[str]],
):
    """Create data folds with unseen arrangements from seen users."""
    print("Generating folds for in-distribution evaluation...")
    test_labels_dict = {}
    val_label_dict = {}
    for user in user_list:
        arrangement_labels = filtered_arrangement_dict[user]
        test_labels = arrangement_labels.copy()
        non_test_labels = [
            [x for x in arrangement_labels if x != tl] for tl in test_labels
        ]  # All arrangement labels not belonging to test for each fold.
        # Randomly select an observation and goal label to be in the validation set.
        val_label_dict[user] = [random.choice(y) for y in non_test_labels]
        if any(not l for l in val_label_dict[user]):
            raise ValueError(f"User {user} has no validation data.")
        test_labels_dict[user] = test_labels

    fold_list = [{m: pd.DataFrame(columns=PD_COLS) for m in MODES} for _ in range(6)]
    for key, data in scene_metadata.items():
        if data["user_id"] not in user_list:
            continue
        placed = False
        for fold_id in range(6):
            if fold_id >= len(test_labels_dict[data["user_id"]]):
                continue
            test_label = test_labels_dict[data["user_id"]][fold_id]
            # Test label matches with goal labels.
            if data["goal_label"] == test_label:
                assert all(
                    x != test_label for x in data["demonstration_labels"]
                ), "Goal label is in demonstration labels."
                fold_list[fold_id]["test"] = _append_dict_to_pd(
                    key, data, fold_list[fold_id]["test"]
                )
                placed = True
            # Test label not matching with goal and demonstrations labels.
            elif (
                all(x != test_label for x in data["demonstration_labels"])
                and data["goal_label"] != test_label
            ):
                val_label = val_label_dict[data["user_id"]][fold_id]
                # Goal label matches with validation label.
                if data["goal_label"] == val_label:
                    fold_list[fold_id]["val"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["val"]
                    )
                # Goal and demo labels do not match with validation label.
                elif (
                    data["goal_label"] != val_label
                    and val_label not in data["demonstration_labels"]
                ):
                    fold_list[fold_id]["train"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["train"]
                    )
                placed = True
        if not placed:
            raise ValueError(f"Scene {key} not placed in any fold.")
    return fold_list, test_labels_dict, val_label_dict


def fold_data_out_distribution(
    scene_metadata: Dict[str, str],
    eligible_users: List[str],
    user_metadata: Dict[str, Dict[str, str]],
):
    """Create data folds to evaluate novel (untrained) user preferences.

    The function generates 5 folds, with each fold witholding one user per
    container-household pair.
    """
    print("Generating folds for novel user evaluation...")
    test_users_per_fold = list([] for _ in range(5))
    for ctype in constants.ENVIRONMENTS.ENVIRONMENT_LIST:
        for household in range(1, 4):
            user_list_ch = list(
                set(
                    user
                    for user, val in user_metadata.items()
                    if val["container_type"] == ctype
                    and int(val["household"]) == household
                )
            )  # 4-5 users per ch pair.
            if not user_list_ch:
                print(f"No users for {ctype}/{household}.")
                continue
            print(f"{len(user_list_ch)} users for {ctype}/{household}")
            for i, user in enumerate(user_list_ch):
                test_users_per_fold[i].append(user)

    val_users_per_fold = list(None for _ in range(5))
    fold_list = [{m: pd.DataFrame(columns=PD_COLS) for m in MODES} for _ in range(5)]
    for key, data in scene_metadata.items():
        if data["user_id"] not in eligible_users:
            raise ValueError(f"User {data['user_id']} not in eligible users.")
        for fold_id in range(5):
            if data["user_id"] in test_users_per_fold[fold_id]:
                fold_list[fold_id]["test"] = _append_dict_to_pd(
                    key, data, fold_list[fold_id]["test"]
                )
            else:
                # Assign a single user for this fold's validation data.
                if val_users_per_fold[fold_id] is None:
                    val_users_per_fold[fold_id] = data["user_id"]
                # Assign scene to train or val based on user id.
                if data["user_id"] == val_users_per_fold[fold_id]:
                    fold_list[fold_id]["val"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["val"]
                    )
                else:
                    fold_list[fold_id]["train"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["train"]
                    )
    return fold_list, test_users_per_fold, val_users_per_fold


def fold_data_unseen_cat(
    scene_metadata: Dict[str, str],
    eligible_users: List[str],
    user_metadata: Dict[str, Dict[str, str]],
):
    """Create data folds to evaluate novel env category preferences.
    """
    print("Generating folds for novel environment category evaluation...")
    test_users_per_fold = list([] for _ in range(len(constants.CATEGORIES.CATEGORY_LIST)))
    for i, cat_dict in enumerate(constants.CATEGORIES.CATEGORY_LIST):
        print(f"Category {list('ABC')[i]}")
        for ctype, household_list in cat_dict.items():
            print(f"Env {ctype}: Variant {household_list}")
            test_user_list = list(
                set(
                    user
                    for user, val in user_metadata.items()
                    if val["container_type"] == ctype
                    and int(val["household"]) in household_list
                )
            )
            test_users_per_fold[i].extend(test_user_list)

    num_folds = len(constants.CATEGORIES.CATEGORY_LIST)
    val_users_per_fold = [None for _ in range(num_folds)]
    fold_list = [{m: pd.DataFrame(columns=PD_COLS) for m in MODES} for _ in range(num_folds)]
    for key, data in scene_metadata.items():
        if data["user_id"] not in eligible_users:
            raise ValueError(f"User {data['user_id']} not in eligible users.")
        for fold_id in range(num_folds):
            if data["user_id"] in test_users_per_fold[fold_id]:
                fold_list[fold_id]["test"] = _append_dict_to_pd(
                    key, data, fold_list[fold_id]["test"]
                )
            else:
                if val_users_per_fold[fold_id] is None:
                    val_users_per_fold[fold_id] = data["user_id"]
                if data["user_id"] == val_users_per_fold[fold_id]:
                    fold_list[fold_id]["val"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["val"]
                    )
                else:
                    fold_list[fold_id]["train"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["train"]
                    )
    assert all(
        v is not None for v in val_users_per_fold
    ), "Some folds don't have validation data."
    return fold_list, test_users_per_fold, val_users_per_fold


def fold_data_unseen_env(
    scene_metadata: Dict[str, str],
    eligible_users: List[str],
    user_metadata: Dict[str, Dict[str, str]],
):
    """Create data folds to evaluate novel (untrained) environment preferences.

    The function generates 3 folds, with each fold witholding one environment
    variant per environment class.
    """
    print("Generating folds for novel environment evaluation...")
    test_users_per_fold = list([] for _ in range(3))
    for household in range(1, 4):
        test_user_list = list(
            set(
                user
                for user, val in user_metadata.items()
                if int(val["household"]) == household
            )
        )
        test_users_per_fold[household - 1].extend(test_user_list)

    val_users_per_fold = [None for _ in range(3)]
    fold_list = [{m: pd.DataFrame(columns=PD_COLS) for m in MODES} for _ in range(3)]
    for key, data in scene_metadata.items():
        if data["user_id"] not in eligible_users:
            raise ValueError(f"User {data['user_id']} not in eligible users.")
        for fold_id in range(3):
            if data["user_id"] in test_users_per_fold[fold_id]:
                fold_list[fold_id]["test"] = _append_dict_to_pd(
                    key, data, fold_list[fold_id]["test"]
                )
            else:
                if val_users_per_fold[fold_id] is None:
                    val_users_per_fold[fold_id] = data["user_id"]
                if data["user_id"] == val_users_per_fold[fold_id]:
                    fold_list[fold_id]["val"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["val"]
                    )
                else:
                    fold_list[fold_id]["train"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["train"]
                    )
    assert all(
        v is not None for v in val_users_per_fold
    ), "Some folds don't have validation data."
    return fold_list, test_users_per_fold, val_users_per_fold


def _split_objects_into_folds(
    object_id_dict: Dict[str, Any], num_folds: int=6
):
    """Split object categories into N folds for testing unseen obj generalization."""
    columns = columns = list(object_id_dict.values())[0].keys()
    object_df = pd.DataFrame(list(object_id_dict.values()), columns=columns)
    test_object_sets = [[] for _ in range(num_folds)]
    for receptacle in object_df.relevant_receptacle.unique():
        objects = object_df[object_df.relevant_receptacle.eq(receptacle)]["fullId"].tolist()
        random.shuffle(objects)  # Shuffle objects for random assignment.
        splits = [objects[i::num_folds] for i in range(num_folds)]
        random.shuffle(splits)  # Shuffle the splits to equalize number of objs across folds.
        for s in splits:
            print("\t",s)
        for i in range(num_folds):
            test_object_sets[i].extend(splits[i])
    return test_object_sets


def fold_data_unseen_obj(
    scene_metadata: Dict[str, str],
    object_id_dict: Dict[str, Any],
    num_folds: int=6
):
    """Create data folds to evaluate unseen object categories."""
    print("Generating folds for unseen object generalization...")
    object_id_list = list(object_id_dict.keys())
    random.shuffle(object_id_list)

    # List of objects used by in each user arrangement.
    arrangement_obj_dict = utils_data.return_objects_used_per_arrangement(
        FLAGS.user_data_dir
    )

    # Create 6 folds of withheld object categories. Also withhold some objects
    # for validation.
    test_objects = _split_objects_into_folds(object_id_dict, num_folds)
    # Split scenes into folds based on the objects used in each example.
    fold_list = [{m: pd.DataFrame(columns=PD_COLS) for m in MODES} for _ in range(num_folds)]
    i = 0
    for key, data in scene_metadata.items():
        if i % 1000 == 0:
            print(f"Processing {i}/{len(scene_metadata)} examples.")
        user_id = data["user_id"]
        all_labels = data["demonstration_labels"] + [data["goal_label"]]
        object_ids_used = []
        for l in all_labels:
            object_ids_used.extend(arrangement_obj_dict[user_id][l])
        for fold_id in range(num_folds):
            test_obj_fold = test_objects[fold_id]
            assert isinstance(test_obj_fold, list)
            if all(oid not in test_obj_fold for oid in object_ids_used):
                # Quick validation split: All goal labels C are in validation.
                if data["goal_label"] == "C":
                    fold_list[fold_id]["val"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["val"]
                    )
                else:
                    fold_list[fold_id]["train"] = _append_dict_to_pd(
                        key, data, fold_list[fold_id]["train"]
                    )
            else:
                fold_list[fold_id]["test"] = _append_dict_to_pd(
                    key, data, fold_list[fold_id]["test"]
                )
        i += 1
    return fold_list, test_objects


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if not FLAGS.in_distribution and not FLAGS.out_distribution and not FLAGS.unseen_env and not FLAGS.unseen_obj and not FLAGS.unseen_cat:
        raise ValueError("No data splits selected. Please select at least one.")

    # Filter user list for those eligible to use in the dataset.
    with open(FLAGS.user_list, "r") as f:
        all_users = f.read().splitlines()
    all_users = list(u for u in all_users if not u.startswith("#"))
    assert (
        len(all_users) == 75
    ), f"Incorrect number of eligible users: {len(all_users)}."
    acq_users = utils_data.find_eligible_users()
    eligible_users = list(set(all_users) & set(acq_users))
    print(f"Number of users eligible for the dataset: {len(eligible_users)}")
    # Load filtered arrangement labels for all users.
    filtered_arrangement_dict = utils_data.return_filtered_arrangement_labels(
        FLAGS.user_data_dir
    )
    assert all(
        u in filtered_arrangement_dict for u in eligible_users
    ), "Some eligible users are missing from the filtered arrangements."

    # Filter scene metadata to only eligible users and valid arrangements.
    dataset_folder = Path(FLAGS.dataset)
    print("Loading and filtering glossary...")
    with open(dataset_folder / "glossary.json", "r") as fjson:
        glossary = json.load(fjson)
    glossary_filtered, user_metadata = _filter_glossary(
        glossary, eligible_users, filtered_arrangement_dict
    )
    print(f"Filtered glossary to {len(glossary)} examples.")

    destination_folder = Path(FLAGS.destination)
    destination_folder.mkdir(exist_ok=True, parents=True)
    print(f"Created {str(destination_folder.absolute())}")

    if FLAGS.in_distribution:
        in_dist_data, test_labels_per_user, val_labels_per_user = fold_data_in_distribution(
            glossary_filtered, eligible_users, filtered_arrangement_dict
        )
        fold_metadata = {
            "test_labels_per_user": test_labels_per_user,
            "val_labels_per_user": val_labels_per_user,
        }
        print("IN-DISTRIBUTION FOLDS")
        _print_fold_statistics(in_dist_data, fold_metadata)
        _save_fold(in_dist_data, fold_metadata, destination_folder / "in_distribution.pkl")

    if FLAGS.out_distribution:
        out_dist_data, test_users, val_users = fold_data_out_distribution(
            glossary_filtered, eligible_users, user_metadata
        )
        fold_metadata = {"test_users": test_users, "val_users": val_users}
        print("OUT-DISTRIBUTION FOLDS")
        _print_fold_statistics(out_dist_data, fold_metadata)
        _save_fold(
            out_dist_data, fold_metadata, destination_folder / "out_distribution.pkl"
        )

    if FLAGS.unseen_env:
        unseen_env_data, test_users, val_users = fold_data_unseen_env(
            glossary_filtered, eligible_users, user_metadata
        )
        fold_metadata = {"test_users": test_users, "val_users": val_users}
        print("UNSEEN ENVIRONMENT FOLDS")
        _print_fold_statistics(unseen_env_data, fold_metadata)
        _save_fold(unseen_env_data, fold_metadata, destination_folder / "unseen_env.pkl")

    if FLAGS.unseen_obj:
        object_id_dict, _ = utils_data.return_object_surface_constants()
        unseen_obj_data, test_objects = fold_data_unseen_obj(
            glossary_filtered, object_id_dict, num_folds=6
        )
        fold_metadata = {"test_objects": test_objects}
        print("UNSEEN OBJECT FOLDS")
        _print_fold_statistics(unseen_obj_data, fold_metadata)
        _save_fold(unseen_obj_data, fold_metadata, destination_folder / "unseen_obj.pkl")

    if FLAGS.unseen_cat:
        unseen_cat_data, test_users, val_users = fold_data_unseen_cat(
            glossary_filtered, eligible_users, user_metadata
        )
        fold_metadata = {"test_users": test_users, "val_users": val_users}
        print("UNSEEN CATEGORY FOLDS")
        _print_fold_statistics(unseen_cat_data, fold_metadata)
        _save_fold(unseen_cat_data, fold_metadata, destination_folder / "unseen_cat.pkl")


if __name__ == "__main__":
    app.run(main)
