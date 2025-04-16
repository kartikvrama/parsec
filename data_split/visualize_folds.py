import sys
import itertools
from typing import List
import json
from pathlib import Path
from absl import app, flags

import random
import multiprocessing as mp
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.append(str(Path("../").absolute()))
matplotlib.rcParams.update({"font.size": 22})

SURF_LABEL_FILE = "../labels/surface_labels_2024_02_29.json"
flags.DEFINE_string("data_dir", None, "Path to the data directory")
flags.DEFINE_string("fold_file", None, "Path to the fold file")
FLAGS = flags.FLAGS

random.seed(42)


def load_file(path):
    if "glossary" in path.name:
        print("Skipping glossary..")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    try:
        result = {
            "user_id": data["user_id"],
            "problem_set": data["problem_set"],
            "container_type": data["container_type"],
            "household": data["household"],
            "num_removed_goal": int(data["num_removed_goal"]),
            "num_demonstrations": int(data["num_demonstrations"]),
            "num_removed_demonstrations": sum(
                int(x) for x in data["num_removed_demonstrations"]
            ),
        }
        if result["num_removed_demonstrations"] + int(data["num_remaining_demonstrations"]):
            result.update({
                "percent_removed_demos": 100*result["num_removed_demonstrations"]/(
                    result["num_removed_demonstrations"] + int(data["num_remaining_demonstrations"])
                    )
            })
        else:
            result.update({"percent_removed_demos": 0})
        return result
    except KeyError as e:
        print(data.keys())
        raise KeyError(e) from e


def return_glossary(key_list: List[str]):
    data_dir = Path(FLAGS.data_dir)
    file_paths = [data_dir / f"{key}.json" for key in key_list]
    with mp.Pool(mp.cpu_count()) as pool:
        pooled_data = pool.map(load_file, file_paths)
    return {
        fp:dt for fp, dt in zip(file_paths, pooled_data)
        if dt is not None
    }


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("Loading folds..")
    with open(FLAGS.fold_file, "r") as fjson:
        fold_data = json.load(fjson)

    pandas_df = None
    for fold_key, fold_dict in fold_data.items():
        print("Fold: ", fold_key)
        mode_keys = fold_dict["test"]
        print(f"Number of examples: {len(mode_keys)}")
        mode_glossary = return_glossary(mode_keys)
        # TODO: get number of objects in train/val/test and overlaps between them.

        users = list(it["user_id"] for it in mode_glossary.values())
        print("Users: ", list((u, users.count(u)) for u in set(users)))

        if pandas_df is None:
            pandas_df = pd.DataFrame.from_dict(mode_glossary, orient="index")
        else:
            pandas_df = pd.concat(
                [pandas_df, pd.DataFrame.from_dict(mode_glossary, orient="index")],
                ignore_index=True
            )

    p = Path("./figs")
    p.mkdir(exist_ok=True, parents=True)
    print(f"Save folder: {p.absolute()}")

    containers = pandas_df["container_type"].unique()
    households = pandas_df["household"].unique()

    for ct, hld in itertools.product(containers, households):
        sub_df = pandas_df[
            (pandas_df["container_type"] == ct) & (pandas_df["household"] == hld)
        ]
        if sub_df is None or len(sub_df) == 0:
            continue
        print(f"Container: {ct}, Household: {hld}")
        _, ax = plt.subplots(2, 2, figsize=(80,80))
        s1 = sns.countplot(sub_df, x="num_demonstrations", hue="user_id", ax=ax[0][0])
        s1.yaxis.set_major_locator(ticker.MultipleLocator(250))
        ax[0][0].set_title("Bar: Number of demos")

        s2 = sns.countplot(sub_df, x="num_removed_goal", hue="user_id", ax=ax[0][1])
        s2.yaxis.set_major_locator(ticker.MultipleLocator(250))
        ax[0][1].set_title("Bar: Num removed goal")

        s3 = sns.countplot(sub_df, x="num_removed_demonstrations", hue="user_id", ax=ax[1][0])
        s3.yaxis.set_major_locator(ticker.MultipleLocator(250))
        ax[1][0].set_title("Hist: Num removed demonstrations")

        s4 = sns.histplot(
            sub_df, x="percent_removed_demos", hue="user_id", ax=ax[1][1],
            binwidth=10, multiple="dodge"
        )
        s4.yaxis.set_major_locator(ticker.MultipleLocator(250))
        ax[1][1].set_title("Hist: Percentage of removed from demonstrations")

        plt.savefig(f"./figs/{ct}_{hld}_vizualization.png")
        plt.close()


if __name__ == "__main__":
    app.run(main)
