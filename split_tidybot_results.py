"""Generates upper-bound and random baseline results for Tidybot model."""

from datetime import datetime
from pathlib import Path
import random
import pickle as pkl
import pandas as pd
from absl import app, flags

flags.DEFINE_string("fold", None, "Fold")
flags.DEFINE_string(
    "input_result_folder", None, "Result folder for original tidybot model."
)
flags.DEFINE_string("model_tag", None, "Model tag to save results.")
FLAGS = flags.FLAGS

DATE_TAG = None


def _create_result_folder(result_folder_parent):
    result_folder_parent = Path(result_folder_parent)
    if result_folder_parent.exists():
        print(
            f"WARNING: folder {result_folder_parent} already exists. Creating a new results folder."
        )
        result_folder_parent = Path(f"{result_folder_parent}-{DATE_TAG}")
    result_folder_parent.mkdir(parents=True)
    return result_folder_parent


def _append_to_pd(dataframe, new_dict):
    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame.from_dict(
                {k: [v] for k, v in new_dict.items()}, orient="columns"
            ),
        ],
        ignore_index=True,
    )
    return dataframe


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    global DATE_TAG
    with open(FLAGS.fold, "rb") as fp:
        fold_dict_parent = pkl.load(fp)
    num_folds = len(fold_dict_parent) - 1

    DATE_TAG = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if FLAGS.model_tag is None:
        raise ValueError("Please specify model tag.")

    results_folder_parent_ub = _create_result_folder(
        f"results/tidybot-UB-{FLAGS.model_tag}"
    )
    results_folder_parent_random = _create_result_folder(
        f"results/tidybot-random-{FLAGS.model_tag}"
    )
    for fkey in list(f"fold_{i}" for i in range(1, 1 + num_folds)):
        df_best = pd.DataFrame()
        df_random = pd.DataFrame()
        subf = Path(FLAGS.input_result_folder) / fkey
        df_original = pd.read_csv(subf / "results.csv")
        df_original["user"] = df_original["scene_id"].apply(
            lambda x: x.split("_", maxsplit=1)[0]
        )
        df_original["goal_label"] = df_original["scene_id"].apply(
            lambda x: x.split("_", maxsplit=1)[1][0]
        )
        df_original["observed_label"] = df_original["scene_id"].apply(
            lambda x: x.split("_")[5][5]
        )
        for user in df_original.user.unique():
            user_df = df_original[df_original.user.eq(user)]
            for goal_label in user_df.goal_label.unique():
                user_glabel_df = user_df[user_df.goal_label.eq(goal_label)]
                if user_glabel_df.empty:
                    print(f"{goal_label} not found for {user}")
                    continue
                user_glabel_df = user_glabel_df.sort_values(
                    "edit_distance", ascending=True
                )
                assert (
                    user_glabel_df.iloc[0].edit_distance
                    == user_glabel_df.edit_distance.min()
                )
                assert (
                    user_glabel_df.iloc[-1].edit_distance
                    == user_glabel_df.edit_distance.max()
                )
                df_best = _append_to_pd(df_best, user_glabel_df.iloc[0].to_dict())
                random_index = random.randint(0, len(user_glabel_df) - 1)
                df_random = _append_to_pd(
                    df_random, user_glabel_df.iloc[random_index].to_dict()
                )

        # Save results for upper bound and random baselines per fold.
        result_folder_fold_ub = results_folder_parent_ub / fkey
        result_folder_fold_ub.mkdir(exist_ok=False)
        df_best.to_csv(result_folder_fold_ub / "results.csv", index=False)

        result_folder_fold_rd = results_folder_parent_random / fkey
        result_folder_fold_rd.mkdir(exist_ok=False)
        df_random.to_csv(result_folder_fold_rd / "results.csv", index=False)
        print(f"Results saved for {fkey}.")


if __name__ == "__main__":
    app.run(main)
