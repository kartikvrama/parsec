"""Script to generate json problem sets from object scenes per receptacle.

This script generates as many problem sets as there are scenes per
receptacle.

Arguments:
    pkl_file: Pickle file of problem sets.
"""

import json
from datetime import datetime
from pathlib import Path
import random
import pickle as pkl
from sys import argv

from generate_object_combinations import IDENTIFIER_FIELD, NAME_FIELD

random.seed(42)

#TODO: Mention a destination folder.
# Constants.
BATCH_SIZE = 6 # Number of scenes per problem set.
RECEPT_TO_HTML = {
    "bathroom_cabinet": "BathroomCabinet",
    "kitchen_cabinet": "KitchenCabinet",
    "chest_of_drawers": "DresserDrawer",
    "shelving_unit": "Bookshelf",
    "fridge": "Fridge",
}

def load_object_samples(file_path: str):
    if not file_path.split("/")[-1].startswith("object_combinations"):
        raise ValueError(
            f"File {file_path} should be of the form 'object_combinations_*.pkl'."
        )
    with open(file_path, "rb") as fpickle:
        data = pkl.load(fpickle)
    return data


def main():
    if len(argv) != 2:
        raise ValueError("Please provide only one pkl file.")
    pkl_file = argv[1]  # List of all pkl files to combine.

    date_time_obj = datetime.now()
    date_time_stamp = date_time_obj.strftime("%Y-%m-%d_%H-%M-%S")

    data = load_object_samples(pkl_file)
    receptacles_list = list(data["objects_by_recept"].keys())

    # Sample batches of problem sets from each pkl file.
    batch_dict = {RECEPT_TO_HTML[k]: {} for k in receptacles_list}
    shuffle_indices = list(range(len(data["problems_by_recept"][receptacles_list[0]])))
    random.shuffle(shuffle_indices) # Shuffle the scenes.
    for recept in receptacles_list:
        problem_list = data["problems_by_recept"][recept]
        num_scenes = len(problem_list)
        for i in range(0, num_scenes, BATCH_SIZE):
            problem_sample_list = [
                {
                    f"object_{j}": {
                            "id": elem[IDENTIFIER_FIELD],
                            "name": elem[NAME_FIELD],
                    }
                    for j, elem in enumerate(problem["objects"])
                } for problem in problem_list[i:i+BATCH_SIZE]
            ]
            problem_sample_dict = {
                "file_name": Path(pkl_file).name,
                "indices": shuffle_indices[i:i+BATCH_SIZE],
                "problem_id": i//BATCH_SIZE, "receptacle": recept,
                "problem_list": problem_sample_list
            }
            batch_dict[RECEPT_TO_HTML[recept]][f"batch_{i//BATCH_SIZE}"] = problem_sample_dict

    with open(f"problem_sets/problem_sets_{date_time_stamp}.json", "w") as fjson:
        json.dump(batch_dict, fjson, indent=4)


if __name__ == "__main__":
    main()
