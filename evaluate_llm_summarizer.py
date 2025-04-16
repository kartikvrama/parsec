"""Script to evaluate TidyBot responses."""
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import pickle as pkl
from absl import app, flags
from utils import utils_data
from utils import utils_eval

flags.DEFINE_string("dataset", None, "Path to the dataset folder.")
flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_string(
    "responses", None, "Path to folder containing Tidybot responses."
)
flags.DEFINE_string("model_tag", None, "Model tag to save results.")
FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    date_tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_tag = FLAGS.model_tag
    if model_tag is None:
        raise ValueError("Please specify model tag.")

    # Create a group folder for saving results.
    result_folder_parent = Path(f"results/tidybot_plus-{model_tag}")
    if result_folder_parent.exists():
        print(
            f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder."
        )
        result_folder_parent = Path(f"results/tidybot-{FLAGS.model_tag}-{date_tag}")
    result_folder_parent.mkdir(parents=True)

    # Load dataset and folds.
    if FLAGS.dataset is None:
        raise ValueError("Please specify dataset folder.")
    dataset_folder = Path(FLAGS.dataset)
    if not dataset_folder.exists():
        raise ValueError("Please specify dataset folder.")
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    # Check that the response folder exists.
    response_folder = Path(FLAGS.responses)
    if not response_folder.exists():
        raise ValueError("Please specify valid response folder.")

    # Evaluation
    for fkey, fold_df_dict in fold_dict.items():
        result_array = []
        prediction_array = {}
        if fkey == "metadata":
            continue
        print(f"Evaluating on {fkey}.")

        # Filter examples with only one demonstration and empty initial scenes.
        test_df_filtered = fold_df_dict["test"]
        test_df_filtered = utils_data.return_fold_max_observations(
            test_df_filtered,
            test_df_filtered["user_id"].unique().tolist(),  # TODO: is there a better way to obtain the user list?
            None,
            None,
        )

        with open(dataset_folder / fkey / "test.pickle", "rb") as fp:
            test_batches = {
                key: val for key, val in pkl.load(fp).items()
                if key in test_df_filtered["scene_id"].tolist()
            }
        print(f"Number of test examples: {len(test_batches)}")

        for scene_id, scene_dict in test_batches.items():
            if not (response_folder / f"{scene_id}.pkl").exists():
                raise ValueError(f"Response file {scene_id}.pkl not found.")

            try:
                with open(response_folder / f"{scene_id}.pkl", "rb") as fpkl:
                    response_dict = pkl.load(fpkl)
            except UnicodeDecodeError as e:
                print(f"Error reading {scene_id}.pkl. Error: {e}.")
                continue

            # Generate and evaluate predicted object configurationfrom responses.
            predicted_scene = response_dict["predicted"]
            partial_scene = scene_dict["partial"]
            goal_scene = scene_dict["goal"]

            edit_distance, ipc, igo = utils_eval.compute_eval_metrics(
                predicted_scene, goal_scene
            )
            assert edit_distance == response_dict["edit_distance"]

            # Save evaluation metrics.
            unplaced_object_names = list(
                o.name
                for o in scene_dict["partial"][-1].objects_on_surface
            )
            num_misplaced_object_classes = len(set(unplaced_object_names))
            rdict = {
                "scene_id": scene_id,
                "container_type": scene_dict["container_type"],
                "household": scene_dict["household"],
                "num_demonstrations": scene_dict["num_demonstrations"],
                "num_removed_demonstrations": sum(
                    int(x)
                    for x in scene_dict["num_removed_demonstrations"]
                ),
                "num_misplaced_objects": len(unplaced_object_names),
                "num_misplaced_object_classes": num_misplaced_object_classes,
                "edit_distance": edit_distance,
                "igo": igo,
                "ipc": ipc,
            }
            result_array.append(rdict)
            prediction_array[scene_id] = {
                "demonstration_scenes": [
                    deepcopy(x)
                    for x in scene_dict["observed_arrangement_list"]
                ],
                "partial_scene": partial_scene,
                "goal_scene": goal_scene,
                "predicted_scene": predicted_scene,
            }

        # Create the folder to save results.
        result_folder = result_folder_parent / fkey
        result_folder.mkdir(exist_ok=False)

        # Save results
        with open(result_folder / "results.csv", "w") as fcsv:
            csv_writer = csv.DictWriter(fcsv, fieldnames=result_array[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(result_array)
        with open(result_folder / "predictions.pkl", "wb") as fpkl:
            pkl.dump(prediction_array, fpkl)

    # Save metadata.
    eval_metadata = {
        "dataset_folder": FLAGS.dataset,
        "response_folder": FLAGS.responses,
    }
    with open(result_folder / "evaluation_metadata.json", "w") as fjson:
        json.dump(eval_metadata, fjson, indent=4, sort_keys=True)


if __name__ == "__main__":
    app.run(main)
