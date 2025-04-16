"""Script to generate TidyBot-Random prompts from permuted user data."""
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from absl import app, flags
import pickle as pkl

sys.path.append(str(Path("../").absolute()))
from utils import constants
from utils import utils_data
from utils import utils_eval
from tidybot_core import utils_tidybot
from tidybot_core import prompt_template_tidybot as prompt_template

flags.DEFINE_string(
    "dataset", None, "Path to folder containing permuted examples.",
)
flags.DEFINE_string(
    "fold", None, "PKL file containing data folds."
)
flags.DEFINE_string(
    "destination_folder", None, "Path to destination folder for llm prompts."
)
FLAGS = flags.FLAGS

# Constants and global definitions.
SURF_CONST = None
USER_OBS_LABEL = None
USER_OBS_SCENE = None
random.seed(constants.SEED)


def example_to_prompt(
    file_path: Path, template_summary: str, template_placement: str
):
    """Wraps around the scene_to_prompt function to generate prompt templates."""
    with open(file_path, "r") as fjson:
        data = json.load(fjson)

    user_id = data["user_id"]
    partial_scene = utils_data.json_to_scene(
        data["partial_scene"], surface_constants=SURF_CONST
    )
    goal_label = data["goal_label"]
    observed_scene_labels = data["demonstration_labels"]

    if user_id not in USER_OBS_LABEL:
        assert user_id not in USER_OBS_SCENE
        USER_OBS_LABEL[user_id] = {}
        USER_OBS_SCENE[user_id] = {}
    if goal_label in USER_OBS_LABEL[user_id]:
        assert goal_label in USER_OBS_SCENE[user_id]
        index = observed_scene_labels.index(USER_OBS_LABEL[user_id][goal_label])
        observed_scene = utils_data.json_to_scene(
            data["demonstration_scenes"][index],
            surface_constants=SURF_CONST
        )
        sed, _ = utils_eval.calculate_edit_distance(
            observed_scene, USER_OBS_SCENE[user_id][goal_label]
        )
        if sed > 0:
            raise ValueError("Observed scene does not match the scene stored in cache.")
    else:
        assert goal_label not in USER_OBS_SCENE[user_id]
        random_label = random.choice(observed_scene_labels)
        observed_scene = utils_data.json_to_scene(
            data["demonstration_scenes"][observed_scene_labels.index(random_label)],
            surface_constants=SURF_CONST,
        )
        USER_OBS_LABEL[user_id][goal_label] = random_label
        USER_OBS_SCENE[user_id][goal_label] = observed_scene

    # Generate the summary prompt using the demo scene.
    summary_prompt = utils_tidybot.scene_to_summary_prompt(observed_scene, summary=None)
    summary_prompt = summary_prompt.replace(constants.TEMPLATE_PH, template_summary)
    # Generate object placements using the parital scene.
    placement_prompt, unplaced_object_list = utils_tidybot.scene_to_placement_prompt(partial_scene)
    placement_prompt = placement_prompt.replace(constants.TEMPLATE_PH, template_placement)
    return summary_prompt, placement_prompt, unplaced_object_list


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    global SURF_CONST
    global USER_OBS_LABEL
    global USER_OBS_SCENE

    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _, SURF_CONST = utils_data.return_object_surface_constants()

    # Load glossary of examples.
    dataset_folder = Path(FLAGS.dataset)
    fold_path = Path(FLAGS.fold)
    with open(fold_path, "rb") as fpkl:
        data_folds = pkl.load(fpkl)

    destination_folder = Path(FLAGS.destination_folder)
    destination_folder.mkdir(exist_ok=True, parents=True)
    metadata = {
        "dataset_folder": FLAGS.dataset,
        "fold_file": FLAGS.fold,
        "date_time_str": date_time_str,
        "summary_template": prompt_template.TIDYBOT_SUMMARY_TEMPLATE,
        "placement_template": prompt_template.TIDYBOT_PLACEMENT_TEMPLATE
    }

    # Nested dictionary of which observed arrangement to use per user per goal label.
    USER_OBS_LABEL = {}
    USER_OBS_SCENE = {}

    dataset_keys = []
    for fkey, files in data_folds.items():
        if fkey == "metadata":
            continue
        data_keys_f = []
        print(f"Generating prompts for {fkey}...")
        save_folder = destination_folder / fkey
        save_folder.mkdir(exist_ok=True, parents=True)
        test_df_filtered = files["test"]
        test_df_filtered = utils_data.return_fold_max_observations(
            test_df_filtered,
            test_df_filtered["user_id"].unique().tolist(),  # TODO: is there a better way to obtain the user list?
            None,
            None,
        )
        for file_name in test_df_filtered["scene_id"].tolist():
            file_path = (dataset_folder / f"{file_name}.json").absolute()
            summary_prompt, placement_prompt, unplaced_object_list = example_to_prompt(
                file_path, template_summary=prompt_template.TIDYBOT_SUMMARY_TEMPLATE,
                template_placement=prompt_template.TIDYBOT_PLACEMENT_TEMPLATE
            )
            # Save prompts and list of unplaced objects to file.
            data_dict = {
                "key": file_name,
                "permuted_example_path": str(file_path),
                "summary_prompt": summary_prompt,
                "placement_prompt": placement_prompt,
                "unplaced_objects": unplaced_object_list  # For evaluation.
            }
            if (save_folder / f"{file_name}.json").exists():
                print(f"Skipping {file_name}.")
                continue
            with open(save_folder / f"{file_name}.json", "w") as fjson:
                json.dump(data_dict, fjson, indent=2)
            data_keys_f.append(file_name)
        print(f"Number of examples for {fkey}: {len(data_keys_f)}")
        dataset_keys.extend(data_keys_f)
    print(f"Total number of examples: {len(dataset_keys)}")

    # Save/overwrite metadata.
    metadata.update({
        "dataset_keys": dataset_keys,
        "user_observed_label_dict": USER_OBS_LABEL
    })
    with open(destination_folder / "metadata.json", "w") as fjson:
        json.dump(metadata, fjson, indent=2)


if __name__ == "__main__":
    app.run(main)
