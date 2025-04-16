import sys
import json
from pathlib import Path
import random
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

sys.path.append(str(Path("../").absolute()))
from utils import utils_data


SURF_LABEL_FILE = "../labels/surface_labels_2024_02_29.json" # Surface constants.

def main():
    with open(SURF_LABEL_FILE, "r") as fjson:
        surface_constants = json.load(fjson)

    dataset_folder = Path("/coc/flash5/kvr6/data/declutter_user_data/permuted_examples_mar26/")
    llm_prompt_folder = Path("/coc/flash5/kvr6/data/declutter_user_data/llm_prompts_april1/prompt_tidybot_template/fold_1")

    keys = llm_prompt_folder.glob("*.json")
    df_master = None
    for key in keys:
        with open(dataset_folder / key.name, "r") as f:
            example = json.load(f)
        df = pd.DataFrame.from_dict({
                "num_demonstrations": [int(example["num_demonstrations"])],
                "num_removed_goal": [int(example["num_removed_goal"])],
                "num_remaining_partial": [int(example["num_remaining_partial"])],
                "goal_label": [example["goal_label"]],
                "demonstration_label": [example["demonstration_labels"][0]],
                "num_removed_demonstrations": [sum(
                    [int(x) for x in example["num_removed_demonstrations"]]
                )],
                "user_id": [example["user_id"]],
            }, orient="columns"
        )
        if df_master is None:
            df_master = df.copy()
        else:
            df_master = pd.concat([df_master, df], ignore_index=True)

        if random.random() > 0.1:
            continue
        print("\n\n----")
        print(key.name)
        print("Demo")
        utils_data.visualize_scene(utils_data.json_to_scene(example["demonstration_scenes"][0], surface_constants))
        print("Partial")
        utils_data.visualize_scene(utils_data.json_to_scene(example["partial_scene"], surface_constants))
        with open(llm_prompt_folder / key.name, "r") as f2:
            prompt_dict = json.load(f2)
        print("Summary_prompt:\n<S>"+prompt_dict["summary_prompt"]+"<E>\n**")
        print("Placement_prompt:\n<S>"+prompt_dict["placement_prompt"]+"<E>\n**")
        print("Unplaced obj list:\n", prompt_dict["unplaced_objects"])

    print(df_master.head())
    # Plot statistics from df.
    for user_id in set(df_master["user_id"]):
        for nd in set(df_master["num_demonstrations"]):
            for gl in set(df_master["goal_label"]):
                for dl in set(df_master["demonstration_label"]):
                    filtered_df = df_master[
                        (df_master["user_id"] == user_id) &
                        (df_master["num_demonstrations"] == nd) &
                        (df_master["goal_label"] == gl) &
                        (df_master["demonstration_label"] == dl)
                    ]
                    if len(filtered_df) > 0:
                        print(f"User: {user_id}, Num_demos: {nd}, Goal label: {gl}, Demo label: {dl}")
                        print(len(filtered_df))
                        fig, ax = plt.subplots(1, 3, figsize=(18, 9))
                        h1 = sns.histplot(data=filtered_df, x="num_removed_demonstrations", kde=True, ax=ax[0], binwidth=1)
                        h1.xaxis.set_major_locator(ticker.MultipleLocator(1))
                        h1.yaxis.set_major_locator(ticker.MultipleLocator(4))
                        ax[0].set_xlabel("Number of removed demonstrations")
                        h2 = sns.histplot(data=filtered_df, x="num_removed_goal", kde=True, ax=ax[1], binwidth=1)
                        h2.xaxis.set_major_locator(ticker.MultipleLocator(1))
                        h2.yaxis.set_major_locator(ticker.MultipleLocator(4))
                        ax[1].set_xlabel("Number of removed goals")
                        h3 = sns.histplot(data=filtered_df, x="num_remaining_partial", kde=True, ax=ax[2], binwidth=1)
                        h3.xaxis.set_major_locator(ticker.MultipleLocator(1))
                        h3.yaxis.set_major_locator(ticker.MultipleLocator(4))
                        ax[2].set_xlabel("Number of remaining goals")
                        plt.savefig(f"./figs/viz_prompts/R-{user_id}_{nd}_{gl}_{dl}_vizualization.png")
                        plt.close()


if __name__ == "__main__":
    main()
