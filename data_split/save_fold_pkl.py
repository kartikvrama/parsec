"""Save pickle batches for training, validation, and testing."""
import sys
from typing import Any, Dict, List
import json
import random
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import pickle as pkl
from absl import app, flags

parent_path = Path("../")
sys.path.append(str(parent_path.absolute()))
from utils import data_struct
from utils import utils_data

flags.DEFINE_string(
    "dataset", None, "Path to the dataset folder containing permuted scenes.",
)
flags.DEFINE_string(
    "fold", None, "Path to pickle file containing data folds."
)
flags.DEFINE_string(
    "destination", None, "Destination folder to save data in.",
)
FLAGS = flags.FLAGS

# Constants.
random.seed(42)
QUEUE_SIZE = 128
SURFACE_CONSTANTS = None


def save_batch(batch: Dict[str, Any], num: int, save_folder: Path):
    """Save JSON batches to disk."""
    assert not (save_folder / f"batch_{num}.pt").exists()
    with open(save_folder / f"batch_{num}.pkl", "wb") as f:
        pkl.dump(batch, f)


def json_to_struct(scene_dict: Dict[str, Any]) -> List[str]:
    """Return the set of objects in the evaluation and goal scenes."""
    object_set = set()
    observed_arrangement_list = list(
        utils_data.json_to_scene(s, SURFACE_CONSTANTS)
        for s in scene_dict["demonstration_scenes"]
    )
    goal = utils_data.json_to_scene(
            scene_dict["goal_scene"], SURFACE_CONSTANTS
    )
    partial = utils_data.json_to_scene(
            scene_dict["partial_scene"], SURFACE_CONSTANTS
    )
    for observed_scene in observed_arrangement_list:
        for surf in observed_scene:
            object_set.update([o.object_id for o in surf.objects_on_surface])
    for surf in goal:
        object_set.update([o.object_id for o in surf.objects_on_surface])
    struct_dict = {
        "observed_arrangement_list": observed_arrangement_list,
        "partial": partial,
        "goal": goal,
    }
    struct_dict.update({
        k: v for k, v in scene_dict.items() if k not in [
            "demonstration_scenes", "goal_scene", "partial_scene"
        ]
    })
    # TODO: Fix this bug in data permutation.
    struct_dict["total_objects_partial"] = 1 + struct_dict["total_objects_partial"]
    assert struct_dict["total_objects_partial"] == struct_dict["num_removed_goal"] + struct_dict["num_remaining_partial"], f"{struct_dict['total_objects_partial']} != {struct_dict['num_removed_goal']} + {struct_dict['num_remaining_partial']}"
    return struct_dict, object_set


def save_dataset(
    file_names: List[str], save_folder: str, save_name: str,
    bool_filter: bool = False
):
    """Generate batches of tensor data from json data after optinally filtering examples.
    
    If bool_filter, only examples with all objects removed will be saved. This is
    reserved for training data to facilitate scene permutation while training.
    """
    object_set = set()
    save_path = save_folder / f"{save_name}.pickle"
    if save_path.exists():
        print(f"Skipping {save_name}.")
        return
    dataset_folder = Path(FLAGS.dataset)
    dataset_dict = {}
    key_list = []
    print(f"Filtering: {bool_filter}")
    for i, fname in enumerate(file_names):
        key = fname.split(".", maxsplit=1)[0]
        with open(dataset_folder / fname, "r") as f:
            example_dict = json.load(f)
        struct_dict, struct_objects = json_to_struct(example_dict)
        if i % 1000 == 0:
            print(f"Processing {i}/{len(file_names)}: Found {len(key_list)} examples.")
        if (struct_dict["num_remaining_partial"] == 0 and bool_filter) or not bool_filter:
            dataset_dict[key] = struct_dict
            key_list.append(key)
            object_set.update(struct_objects)
    with open(save_path, "wb") as f:
        pkl.dump(dataset_dict, f, pkl.HIGHEST_PROTOCOL)
    print(f"Number of final examples: {len(key_list)}")
    return key_list, object_set


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    global SURFACE_CONSTANTS
    _, SURFACE_CONSTANTS = utils_data.return_object_surface_constants()

    print("Generating reduced dataset for debugging.")
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    fold_path = Path(FLAGS.fold)
    with open(fold_path, "rb") as fpkl:
        data_folds = pkl.load(fpkl)

    print(f"Dataset: {FLAGS.dataset}")
    print(f"Fold type: {fold_path.name}")

    for fkey, fold in data_folds.items():
        if fkey == "metadata":
            continue
        print(f"Generating batches for {fkey}...")
        metadata = {
            "dataset_folder": FLAGS.dataset,
            "date_time_str": date_time_str,
            "data_fold_path": str(fold_path.absolute()),
            "data_fold_type": fold_path.name.split(".")[0],
            "fold": fkey
        }
        destination_folder = (
            Path(FLAGS.destination) / f"{fold_path.name.split('.')[0]}" / fkey
        )
        destination_folder.mkdir(exist_ok=True, parents=True)
        for mode, mode_df in fold.items():
            bool_filter = mode == "train"
            # Save all examples.
            mode_keys = mode_df["scene_id"].tolist()
            file_names = list(f"{key}.json" for key in mode_keys)
            batch_keys, object_set = save_dataset(
                file_names, save_folder=destination_folder, save_name=mode,
                bool_filter=bool_filter
            )
            print(f"Number of unique objects in {mode}: {len(object_set)}")
            metadata.update({
                f"{mode}_keys": batch_keys,
                f"{mode}_object_set": object_set
            })
            metadata_path = destination_folder / "metadata.pickle"
            if metadata_path.exists():
                metadata_path = destination_folder / f"metadata_{date_time_str}.pickle"
            # Save metadata.
            with open(metadata_path, "wb") as f:
                pkl.dump(metadata, f, pkl.HIGHEST_PROTOCOL)
        print(f"Done with {fkey}.")
    print("Done.")


if __name__ == "__main__":
    app.run(main)
