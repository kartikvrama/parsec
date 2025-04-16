"""Generate a glossary of all scenes in the permuted data."""
import os
import sys
import json
from pathlib import Path
import pickle as pkl
from absl import app, flags

flags.DEFINE_string(
    "dataset", None, "Directory containing permuted user data."
)
FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    folder = Path(FLAGS.dataset)
    assert folder.exists(), f"Folder {folder} does not exist."

    # Create a glossary of all the objects in the dataset.
    glossary = {}
    for i, file in enumerate(os.scandir(folder)):
        if i % 10000 == 0:
            print(f"Processing {i}th file: {file.name}")
        file_name = file.name
        if file_name.startswith("glossary"):
            continue
        key = file_name.split(".")[0]
        try:
            with open(folder / file.name, "r") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"Error with {file_name}: {e}", file=sys.stderr)
            continue
        glossary.update(
            {
                key: {
                    "file_name": file_name,
                    "user_id": data["user_id"],
                    "problem_set": data["problem_set"],
                    "container_type": data["container_type"],
                    "household": data["household"],
                    "goal_label": data["goal_label"],
                    "num_removed_goal": int(data["num_removed_goal"]),
                    "num_remaining_partial": int(data["num_remaining_partial"]),
                    "demonstration_labels": data["demonstration_labels"],
                    "num_demonstrations": int(data["num_demonstrations"]),
                    "num_removed_demonstrations": list(int(x) for x in data["num_removed_demonstrations"]),
                    "num_remaining_demonstrations": int(data["num_remaining_demonstrations"]),
                }
            }
        )

    with open(folder / "glossary.pickle", "wb") as fpkl:
        pkl.dump(glossary, fpkl, pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app.run(main)
