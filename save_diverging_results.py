"""Plot evaluation metrics for different models."""
from pathlib import Path
from datetime import datetime
from absl import app
from absl import flags
import json
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import constants
from utils.data_struct import SurfaceEntity
from utils import utils_data
from utils import utils_eval
from utils.utils_plot import (
    MODELS,
    MODELS_PLOT,
    COLORS_DISPARATE,
    GROUP_NAMES,
    GROUP_LABEL_DICT,
    find_group,
    load_fold_dict,
    load_model_results,
)

DATA_HOME="/coc/flash5/kvr6/data/declutter_user_data"
flags.DEFINE_string(
    "fold",
    f"{DATA_HOME}/folds-2024-07-20/out_distribution.pkl",
    "Path to the PKL file containing data folds."
)
flags.DEFINE_string(
    "llm_summarizer",
    "results/tidybot_plus-OOD-2024-08-26",
    "Path to the LLM Summarizer results."
)
flags.DEFINE_string(
    "tidybot_random",
    "results/tidybot-OOD-2024-08-23",
    "Path to the TidyBot-Random results."
)
flags.DEFINE_string(
    "consor",
    "results/consor-OOD-2024-07-23-IV",
    "Path to the ConSOR results."
)
flags.DEFINE_string(
    "apricot",
    "results/apricot_noquery-OOD-20250318-latest",
    "Path to the Apricot results."
)
flags.DEFINE_bool(
    "find_users",
    False,
    "Whether to find users with disparate results."
)
FLAGS = flags.FLAGS

matplotlib.rcParams.update({"font.size": 18, "font.weight": "bold", "lines.linewidth": 1})

ENV_VAR_MAPPING = [
    ## Similar-1D
    (constants.ENVIRONMENTS.KITCHEN_CABINET, 2), # kitchen 3x1
    (constants.ENVIRONMENTS.BATHROOM_CABINET, 3), # bathroom 4hs
    (constants.ENVIRONMENTS.BOOKSHELF, 1), # bookshelf 5s
    ## Similar-2D
    (constants.ENVIRONMENTS.KITCHEN_CABINET, 1), # kitchen 3x2
    (constants.ENVIRONMENTS.BATHROOM_CABINET, 2), # bathroom 5os
    ## Dissimilar
    (constants.ENVIRONMENTS.FRIDGE, 2), # fridge 3cs/3dr
    (constants.ENVIRONMENTS.FRIDGE, 3), # fridge 2cs/3ds
]

surface_constants = None

def _return_scene_struct(x):
    global surface_constants
    if not isinstance(x, list):
        return utils_data.json_to_scene(x, surface_constants)
    return x

def _return_json(x):
    if isinstance(x, dict):
        if all(isinstance(v, dict) for v in x.values()):
            return x
    elif all(isinstance(surf, SurfaceEntity) for surf in x):
        return utils_data.scene_to_json(x)
    raise ValueError("Invalid scene format, scene must be in struct or JSON format.")

def _plot_ranks(model_ranking_per_sample):
    date_time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_color_dict = dict(zip(MODELS_PLOT, COLORS_DISPARATE))
    
    ## Use boxplots to show the distribution of ranks
    model_ranking_df = pd.DataFrame(model_ranking_per_sample)
    model_ranking_df = model_ranking_df.sort_values("group")
    # collapse the model ranks into a single column
    model_ranking_df = model_ranking_df.melt(
        id_vars=["scene_id", "group", "num_remaining_partial"],
        value_vars=model_ranking_df.columns[:-2],
        var_name="model",
        value_name="rank"
    )

    fig, axes_array = plt.subplots(1, 3, figsize=(36, 16))
    for i, group in enumerate(GROUP_NAMES):
        sns.barplot(
            x="num_remaining_partial",
            y="rank",
            hue="model",
            data=model_ranking_df[model_ranking_df["group"].eq(group)],
            errorbar="sd",
            legend=True,
            palette=model_color_dict,
            ax=axes_array[i],
        )
        axes_array[i].set_title(f"Group {group}")
    fig.savefig(f"figures/disparate_results_{date_time_str}.png", bbox_inches="tight")

def main(argv):
    global surface_constants
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    ## Constants
    _, surface_constants = utils_data.return_object_surface_constants()
    
    ## Load fold dictionary and metadata
    scene_id_metadata, num_folds = load_fold_dict(FLAGS.fold)
    
    ## Create model folder dictionary
    model_folder_dict = {
        "LLMSummarizer": FLAGS.llm_summarizer,
        "TidyBot-Random": FLAGS.tidybot_random,
        "ConSOR": FLAGS.consor,
        "Apricot": FLAGS.apricot,
    }
    
    ## Load results from all models
    result_df, predictions_per_model, scene_ids_per_model, model_list = load_model_results(
        model_folder_dict, num_folds
    )

    ## Find disparate results and rank models based on edit distance
    user_goal_dict = []
    model_ranking_per_sample = []
    eval_examples_and_predictions = {}
    
    for scene_id in scene_ids_per_model["LLMSummarizer"]:
        filtered_results = [
            result_df[
                (result_df["Model"].eq(model))
                & (result_df["user_id"].eq(scene_id_metadata[scene_id]["user_id"]))
                & (result_df["goal_label"].eq(scene_id_metadata[scene_id]["goal_label"]))
                & (result_df["num_misplaced_objects"].eq(scene_id_metadata[scene_id]["num_removed_goal"]))
            ]
            for model in model_list
        ]
        scene_id_per_model = [x["scene_id"].values[0] for x in filtered_results]
        result_metadata = {
            "container_type": filtered_results[0]["container_type"].values[0],
            "household": filtered_results[0]["household"].values[0],
            "num_misplaced_objects": scene_id_metadata[scene_id]["num_removed_goal"],
            "num_remaining_partial": scene_id_metadata[scene_id]["num_remaining_partial"],
        }
        edit_distances = [x["edit_distance"].values[0] for x in filtered_results]

        if FLAGS.find_users:
            ## Filter: specific container type and household
            if all(x[0]!=result_metadata["container_type"] or x[1]!=result_metadata["household"] for x in ENV_VAR_MAPPING):
                continue
            ## Filter: 2/4/6 objects remaining in the initial scene
            if not int(result_metadata["num_remaining_partial"]) in [2, 4, 6]:
                continue
            ## Filter: reject scenes with more than 7 surfaces
            goal_scene = _return_scene_struct(predictions_per_model[model_list[0]][scene_id_per_model[0]]["goal_scene"])
            if len(goal_scene) > 7:
                continue
            ## Only consider scenes with different model predictions
            if all(x==0 for x in edit_distances):
                continue
            pairwise_edit_distances = [
                utils_eval.calculate_edit_distance(
                    _return_scene_struct(predictions_per_model[m1][s1]["predicted_scene"]),
                    _return_scene_struct(predictions_per_model[m2][s2]["predicted_scene"]),
                )[0]
                for i, (m1, s1) in enumerate(zip(model_list, scene_id_per_model))
                for m2, s2 in zip(model_list[i+1:], scene_id_per_model[i+1:])
            ]
            if any(x==0 for x in pairwise_edit_distances):
                continue
            user_goal_str = f"{scene_id.split('_', maxsplit=1)[0]}_{scene_id_metadata[scene_id]['goal_label']}"
            user_goal_dict.append(
                {
                    "user_goal_str": user_goal_str,
                    "container_type": result_metadata["container_type"],
                    "household": result_metadata["household"],
                    "num_remaining_partial": result_metadata["num_remaining_partial"],
                }
            )
        else:
            with open("results/user_goal_list_filtered.txt", "r") as fp:
                user_goal_list_filtered = [
                    l for l in fp.read().split(",") if l
                ]
            ## Filter: scenes with specific user_id and goal_label
            if not any(
                scene_id.startswith(user_goal_label)
                for user_goal_label in user_goal_list_filtered
            ):
                continue
            ## Filter: 2/4/6 objects remaining in the initial scene
            if not int(result_metadata["num_remaining_partial"]) in [2, 4, 6]:
                continue
            pairwise_edit_distances = [
                utils_eval.calculate_edit_distance(
                    _return_scene_struct(predictions_per_model[m1][s1]["predicted_scene"]),
                    _return_scene_struct(predictions_per_model[m2][s2]["predicted_scene"]),
                )[0]
                for i, (m1, s1) in enumerate(zip(model_list, scene_id_per_model))
                for m2, s2 in zip(model_list[i+1:], scene_id_per_model[i+1:])
            ]
            if any(x==0 for x in pairwise_edit_distances):
                print(f"Skipping {scene_id} because of similar predictions.")
                continue
            eval_examples_and_predictions[scene_id] = {
                "num_remaining_partial": scene_id_metadata[scene_id]["num_remaining_partial"],
                "observed_scenes": [
                    _return_json(scene)
                    for scene in predictions_per_model[model_list[0]][scene_id_per_model[0]]["demonstration_scenes"]
                ],
                "initial_scene": _return_json(
                    predictions_per_model[model_list[0]][scene_id_per_model[0]]["partial_scene"]
                ),
                "container_type": result_metadata["container_type"],
                "household": str(result_metadata["household"]),
            }
            ## Add predicted scenes
            eval_examples_and_predictions[scene_id].update({
                f"{ml}_predicted": _return_json(predictions_per_model[ml][sid_model]["predicted_scene"])
                for ml, sid_model in zip(model_list, scene_id_per_model)
            })
            ## Rank models based on edit distance
            model_ids_sorted = np.argsort(edit_distances)
            model_rank_dict = {model_list[id]: x+1 for x, id in enumerate(model_ids_sorted)}
            model_rank_dict.update({
                "scene_id": scene_id,
                "group": scene_id_metadata[scene_id]["group"],
                "num_remaining_partial": scene_id_metadata[scene_id]["num_remaining_partial"],
            })
            model_ranking_per_sample.append(model_rank_dict)

    if FLAGS.find_users:
        csv_df = pd.DataFrame(user_goal_dict)
        count_df = csv_df.groupby("user_goal_str").count()
        ## get row indexes for which column num_remaining_partial has value 2 or more
        user_goal_list = count_df[count_df.num_remaining_partial.ge(2)].index.tolist()
        user_goal_list_filtered = []
        for ctype, hidx in ENV_VAR_MAPPING:
            ## 2 users per ctype and hidx that belong the filtered_user_goal_list
            temp_df = csv_df[
                csv_df.container_type.eq(ctype)
                & csv_df.household.eq(hidx)
                & csv_df.user_goal_str.isin(user_goal_list)
            ]
            unique_strs_temp = temp_df.user_goal_str.unique()
            count = 0
            for ug in unique_strs_temp:
                user_tag = ug.split("_")[0]
                if not user_goal_list_filtered:
                    user_goal_list_filtered.append(ug)
                    count += 1
                elif any(
                    ug_temp.startswith(user_tag) for ug_temp in user_goal_list_filtered
                ):
                    continue
                else:
                    user_goal_list_filtered.append(ug)
                    count += 1
                if count >= 2:
                    break
            print(f"Container {ctype} & H{hidx}- {count} users")
        ## Save user_goal_list_filtered
        with open("results/user_goal_list_filtered.txt", "w") as fp:
            fp.writelines([f"{x}," for x in user_goal_list_filtered])
    else:
        ## Save eval_examples_and_predictions
        if Path("results/eval_examples_and_predictions.json").exists():
            print("WARNING: Deleting existing eval_examples_and_predictions.json.")
            Path("results/eval_examples_and_predictions.json").unlink()
        with open("results/eval_examples_and_predictions.json", "w") as fp:
            json.dump(eval_examples_and_predictions, fp, indent=4)
        ## Plot model ranks
        print(f"Number of samples with disparate results: {len(model_ranking_per_sample)} out of {len(scene_ids_per_model['LLMSummarizer'])}")
        for group in ["A", "B", "C"]:
            print(f"Group {group}: {len([x for x in model_ranking_per_sample if x['group'] == group])}")
        _plot_ranks(model_ranking_per_sample)


if __name__ == "__main__":
    app.run(main)
