"""Script to load user json data, compute data stats, and generate training examples."""
import os
from typing import Any, Dict
import sys
from pathlib import Path

import csv
import json
import random
from multiprocessing import Pool
from absl import app, flags

current_path = Path('../').absolute()
sys.path.append(str(current_path))
from utils import constants
from utils import utils_data
from data_permutation import utils_data_permute

flags.DEFINE_string(
    "data",
    "/home/kartik/Documents/datasets/declutter_user_data/arrangements_json",
    "Directory containing user data.",
)
flags.DEFINE_string(
    "destination",
    "/home/kartik/Documents/datasets/declutter_user_data/permuted_examples",
    "Destination folder to save data in.",
)
flags.DEFINE_bool(
    "dryrun", False, "Flag to toggle between a dry run and an actual run."
)
flags.DEFINE_bool(
    "verbose", False, "Flag to toggle verbosity of the script."
)
FLAGS = flags.FLAGS

# Constants.
random.seed(constants.SEED)
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
OBJ_LABEL_FILE = "../labels/object_labels_2024_02_29.csv"
SURF_LABEL_FILE = "../labels/surface_labels_2024_02_29.json"
QUEUE_SIZE = 16

# Global variables.
destination_folder = None


def save_json(folder: Path, file_name:str, data: dict[str, Any]):
    """Save data to json file."""
    temp_name = f".{file_name}.tmp"
    with open(folder / temp_name, "w") as fjson:
        json.dump(data, fjson, indent=2)
    os.replace(str(folder/temp_name), str(folder/file_name))


def dry_run(data_dict: Dict[str, Any]):
    """Do a dry run of permuting examples to estimate the size of data generated."""
    print(
        f"Generating training examples for {data_dict['user_id']}, container {data_dict['container_type']}, household {data_dict['household']}"
    )
    num_examples = 0
    for i in range(len(data_dict["arrangement_array"])):
        goal_scene = data_dict["arrangement_array"][i]
        demonstration_scenes = list(data_dict["arrangement_array"][j] for j in range(len(data_dict["arrangement_array"])) if j != i)
        demonstration_labels = list(ALPHABET[j] for j in range(len(data_dict["arrangement_array"])) if j != i)
        permuted_example_generator = utils_data_permute.permute_example(
            goal_scene, ALPHABET[i], demonstration_scenes, demonstration_labels,
            distance_cache={}, user_id=data_dict['user_id']
        )
        first_result = next(permuted_example_generator)
        print(f"Number of scenes this user: {first_result}")
        if first_result is None:
            print(f"Goal scene {ALPHABET[i]} for user {data_dict['user_id']} is empty, skipping.")
            continue
        num_examples += first_result
    return num_examples


def core_permute_example(data_dict: Dict[str, Any]):
    """Generate demonstration-goal permutations of a given example."""
    print(
        f"Generating training examples for {data_dict['user_id']}, container {data_dict['container_type']}, household {data_dict['household']}"
    )
    num_examples = 0
    scene_distance_cache = {}  # Cache scene distances.
    # TODO: we don't need a file queue here anymore.
    file_queue = []
    used_keys = []
    for i in range(len(data_dict["arrangement_array"])):
        if FLAGS.verbose:
            print(f"{data_dict['user_id']}- Goal scene label: {ALPHABET[i]}")
        goal_scene = data_dict["arrangement_array"][i]
        demonstration_scenes = list(data_dict["arrangement_array"][j] for j in range(len(data_dict["arrangement_array"])) if j != i)
        demonstration_labels = list(ALPHABET[j] for j in range(len(data_dict["arrangement_array"])) if j != i)

        # Save permuted examples.
        permuted_example_generator = utils_data_permute.permute_example(
            goal_scene, ALPHABET[i], demonstration_scenes, demonstration_labels,
            scene_distance_cache, user_id=data_dict['user_id']
        )
        
        first_result = next(permuted_example_generator)
        if first_result is None:
            print(f"{data_dict['user_id']}- Goal scene {ALPHABET[i]} is empty, skipping.", file=sys.stderr)
            continue
        for (name, example) in permuted_example_generator:
            new_key = data_dict["user_id"] + "_" + name
            if (destination_folder / f"{new_key}.json").exists():
                print(f"File {new_key}.json already exists, skipping.", file=sys.stderr)
                continue
            assert new_key not in used_keys, f"Key {new_key} already used."
            data = example
            data.update(
                {
                    "user_id": data_dict["user_id"],
                    "problem_set": data_dict["problem_set"],
                    "container_type": data_dict["container_type"],
                    "household": data_dict["household"],
                }
            )
            file_queue.append((destination_folder, f"{new_key}.json", data))
            used_keys.append(new_key)
            num_examples += 1
            if len(file_queue) == QUEUE_SIZE:
                # Write to disk using multiprocessing.
                for (dir, fname, save_data) in file_queue:
                    save_json(dir, fname, save_data)
                file_queue = []
        if FLAGS.verbose:
            print(f"Length of cache: {len(scene_distance_cache)}")
    if file_queue:
        for (dir, fname, save_data) in file_queue:
            save_json(dir, fname, save_data)
    print(f"Number of examples generated for user {data_dict['user_id']}: {num_examples}")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    global destination_folder

    # Load object labels and surface constants.
    object_label_dict = {}
    with open(OBJ_LABEL_FILE, "r") as fcsv:
        csv_data = csv.DictReader(fcsv)
        for row in csv_data:
            object_label_dict[row["fullId"]] = row

    with open(SURF_LABEL_FILE, "r") as fjson:
        surface_constants = json.load(fjson)

    # List all files in data directory.
    user_data_dir = Path(FLAGS.data)
    json_files = list(user_data_dir.glob("*.json"))
    if len(json_files) == 0:
        raise ValueError(f"No json files found in {user_data_dir}.")
    print(f"{len(json_files)} users found in {user_data_dir}.")

    # Filter scenes and store relevant information in array.
    filtered_data = []
    for json_file in json_files:
        json_path = json_file.absolute()
        arrangements, metadata = utils_data.load_user_scene_data(
            json_path, object_label_dict, surface_constants
        )
        # TODO: Filtering function to only load approved scenes.
        data_dict = {
            "user_id": metadata["user_id"],
            "problem_set": metadata["problem_set"],
            "container_type": metadata["container_type"],
            "household": metadata["household"],
            "arrangement_array": arrangements,
        }
        filtered_data.append(data_dict)

    if FLAGS.dryrun:
        # Dry run to calculate number of examples.
        total_num_examples = 0
        examples_per_container = {c:0 for c in constants.ENVIRONMENTS.ENVIRONMENT_LIST}
        for data_dict in filtered_data:
            num_examples = dry_run(data_dict)
            total_num_examples += num_examples
            examples_per_container[data_dict["container_type"]] += num_examples
        print("Number of example per container")
        for c, ct in examples_per_container.items():
            print(f"\t{c}: {ct}")
        print(f"Total: {total_num_examples}")

    else:
        # Generate permuted examples.
        destination_folder = Path(FLAGS.destination)
        destination_folder.mkdir(exist_ok=True, parents=True)
        queue = []
        for household in range(3, 0, -1):
            for ctype in constants.ENVIRONMENTS.ENVIRONMENT_LIST:
                # Permute data by container type.
                data_dict_list = [
                    dd for dd in filtered_data if dd["container_type"] == ctype
                    and dd["household"] == household
                ]
                for data_dict in data_dict_list:
                    queue.append([data_dict])
                    if len(queue) == QUEUE_SIZE:
                        with Pool(QUEUE_SIZE) as p:
                            p.starmap(core_permute_example, queue)
                        queue = []
        if queue:
            with Pool(len(queue)) as p:
                p.starmap(core_permute_example, queue)


if __name__ == "__main__":
    app.run(main)
