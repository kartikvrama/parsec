"""Plot evaluation metrics for different models."""
from pathlib import Path
import pickle as pkl
from absl import app
from absl import flags
import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as spstats

from utils import constants

flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_integer(
    "fold_num",
    None,
    "Fold number to load data from. If None, all folds are loaded."
)
flags.DEFINE_string("llm_summarizer", None, "Path to the LLM Summarizer results.")
flags.DEFINE_string("tidybot_random", None, "Path to the TidyBot-Random results.")
flags.DEFINE_string("neatnet", None, "Path to the TidyBot results.")
flags.DEFINE_string("consor", None, "Path to the ConSOR results.")
flags.DEFINE_string("declutter", None, "Path to the Declutter results.")
flags.DEFINE_string("cf", None, "Path to the CF results.")
flags.DEFINE_string("cffm", None, "Path to the CF results.")
FLAGS = flags.FLAGS

COLORS = ["#bebada", "#fb8072", "#80b1d3", "#fdb462"]
COLORS_DARK = ['#8e8ba3', '#bc6055', '#60849e', '#bd8749']
MODELS = [
    "LLMSummarizer",
    "TENet",
    "TidyBot-Random",
    "ConSOR",
    "NeatNet",
    "CFFM",
    "CF",
]
MODELS_PLOT = [
    "LLMSummarizer",
    "TidyBot-Random",
    "ConSOR",
    "CF",
]
GROUP_LABEL_DICT = [  # Groups are labeled alphabetically (A, B,...).
    # Group A: Identical surface types without left-right symmetry.
    {
        constants.ENVIRONMENTS.KITCHEN_CABINET: [2],
        constants.ENVIRONMENTS.BATHROOM_CABINET: [3],
        constants.ENVIRONMENTS.BOOKSHELF: [1, 2, 3],
    },
    # Group B: Identical surface types with left-right symmetry.
    {
        constants.ENVIRONMENTS.KITCHEN_CABINET: [1, 3],
        constants.ENVIRONMENTS.BATHROOM_CABINET: [2],
    },
    # Group C: > 1 surface types.
    {
        constants.ENVIRONMENTS.BATHROOM_CABINET: [1],
        constants.ENVIRONMENTS.FRIDGE: [1, 2, 3],
        constants.ENVIRONMENTS.VANITY_DRESSER: [1, 2, 3],
    },
]


def _find_group(env, var):
    label_list = list("ABCDEFGHIJKL")[:len(GROUP_LABEL_DICT)]
    for group_id, group_dict in enumerate(GROUP_LABEL_DICT):
        if env in group_dict:
            if var in group_dict[env]:
                return label_list[group_id]
    raise ValueError(f"Could not find group for {env} and {var}.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Load fold dictionary.
    with open(FLAGS.fold, "rb") as fp:
        fold_dict_parent = pkl.load(fp)
    num_folds = len(fold_dict_parent) - 1
    num_remaining_partial_dict = {}
    num_removed_goal_dict = {}
    num_total_goal = {}
    scene_id_to_group = {}
    for fkey, fdf in fold_dict_parent.items():
        if fkey == "metadata":
            continue
        for _, row in fdf["test"].iterrows():
            num_remaining_partial_dict[row["scene_id"]] = int(row["num_remaining_partial"])
            num_removed_goal_dict[row["scene_id"]] = int(row["num_removed_goal"])
            num_total_goal[row["scene_id"]] = int(row["num_remaining_partial"]) + int(row["num_removed_goal"])
            scene_id_to_group[row["scene_id"]] = _find_group(
                row["environment"], row["variant"]
            )

    result_df = None
    for model, folder in zip(
        ["TidyBot-Random", "ConSOR", "TENet", "CF", "LLMSummarizer"],
        [FLAGS.tidybot_random, FLAGS.consor, FLAGS.declutter, FLAGS.cf, FLAGS.llm_summarizer],
    ):
        if folder is None:
            continue
        print(f"Loading results for {model}")
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Results folder {folder} does not exist.")
        for fkey, subf in list((f"fold_{i}", folder / f"fold_{i}") for i in range(1, 1 + num_folds)):
            if FLAGS.fold_num is not None and fkey != f"fold_{FLAGS.fold_num}":
                continue
            elif FLAGS.fold_num is not None:
                print(f"ONLY loading {fkey} results for {model}")
            else:
                print(f"Loading {fkey} results for {model}")
            if not subf.exists():
                raise FileNotFoundError(f"Cannot load {fkey} results for {model}: {subf} does not exist.")
            df = pd.read_csv(subf / "results.csv")
            df["Model"] = model
            if result_df is None:
                result_df = df.copy()
            else:
                result_df = pd.concat([result_df, df], ignore_index=True)

    for model, folder in zip(
        ["NeatNet", "CFFM"],
        [FLAGS.neatnet, FLAGS.cffm],
    ):
        print(f"Loading results for {model}")
        if folder is not None:
            folder = Path(folder)
            if not folder.exists():
                raise FileNotFoundError(f"Results folder {folder} does not exist.")
            for env in constants.ENVIRONMENTS.ENVIRONMENT_LIST:
                for var in range(1, 4):
                    intermediate_folder = folder / env / str(var)
                    if not intermediate_folder.exists():
                        raise FileNotFoundError(f"Cannot load {model} results for {env}/{var}: {intermediate_folder} does not exist.")

                    for fkey, subf in list((f"fold_{i}", intermediate_folder / f"fold_{i}") for i in range(1, 1 + num_folds)):
                        if FLAGS.fold_num is not None and fkey != f"fold_{FLAGS.fold_num}":
                            continue
                        elif FLAGS.fold_num is not None:
                            print(f"ONLY loading {fkey} results for {model}")
                        if not subf.exists():
                            print(f"Skipping {fkey} results for {model}: {subf} does not exist.")
                            continue
                        df = pd.read_csv(subf / "results.csv")
                        df["Model"] = model
                        result_df = pd.concat([result_df, df], ignore_index=True)

    # Add additional columns to the result_df.
    result_df["num_remaining_partial"] = result_df["scene_id"].map(num_remaining_partial_dict)
    result_df["num_removed_goal"] = result_df["scene_id"].map(num_removed_goal_dict)
    result_df["num_total_goal"] = result_df["scene_id"].map(num_total_goal)
    result_df["placement_acc"] = (result_df["num_removed_goal"] - result_df["edit_distance"]) / result_df["num_removed_goal"]
    assert result_df.placement_acc.min() >= 0, f"Negative placement accuracy found: {result_df.placement_acc.min()}"
    result_df["user"] = result_df["scene_id"].apply(lambda x: x.split("_", maxsplit=1)[0])
    result_df["group"] = result_df["scene_id"].map(scene_id_to_group)

    # Check if all models (except ConSOR which needs no observations) have the same scene_ids.
    models = result_df["Model"].unique()
    scene_ids_per_model = result_df.groupby("Model")["scene_id"].unique().to_dict()
    scene_id_sets = [set(scene_ids_per_model[model]) for model in models if model != "ConSOR"]
    assert all(scene_id_sets[0] == scene_id_set for scene_id_set in scene_id_sets), "Scene IDs do not match across models."

    # Sort result_df in specific order of models.
    rdf_filtered = result_df[
        result_df["num_remaining_partial"].le(18)
        & result_df["num_remaining_partial"].apply(lambda x: x % 4 == 0)
        & result_df["Model"].isin(MODELS_PLOT)
    ]

    for metric in ["edit_distance", "igo"]:

        model = ols(
            f"{metric} ~ C(Model) + C(num_remaining_partial) + C(Model):C(num_remaining_partial)",
            data=rdf_filtered
        ).fit()

        _, p_value_normal = sm.stats.diagnostic.kstest_normal(model.resid)
        print(f"{metric}: P value (must be > 0.05) is ", p_value_normal)

        for d in [0, 4, 8, 12, 16]:
            newdf = rdf_filtered[rdf_filtered["num_remaining_partial"].eq(d)]
            # Find arrays of data for each model sorted by scene_ids.
            data = []
            for m in MODELS_PLOT:
                temp = newdf[newdf["Model"].eq(m)]
                temp.sort_values(by="scene_id", inplace=True)
                data.append(temp[metric].values)
            print(f"Number of data points: {len(data[0])}")

            # Friedman test.
            _, p_value = spstats.friedmanchisquare(*data)
            print(f"Friedman test for {metric} at N={d}, p value (< 0.05) {p_value}")
            print("-"*20)

            # Wilcoxon signed-rank post hoc test with Bonferroni correction
            for m1 in range(len(MODELS_PLOT)):
                for m2 in range(m1+1, len(MODELS_PLOT)):
                    result = spstats.wilcoxon(x=data[m1], y=data[m2])
                    p_value = min(1, result.pvalue * 6)
                    stat = result.statistic
                    print(f"{metric} at N={d} between {MODELS_PLOT[m1]} and {MODELS_PLOT[m2]}: p value (< 0.05) {p_value}")
                    if p_value < 0.05:
                        new_result = spstats.wilcoxon(x=data[m1], y=data[m2], alternative="greater")
                        if 6*new_result.pvalue < 0.05:
                            print(f"{MODELS_PLOT[m1]} > {MODELS_PLOT[m2]} with p {6*new_result.pvalue:.3f}\n")
                        else:
                            print(f"{MODELS_PLOT[m1]} < {MODELS_PLOT[m2]}, one sided p {6*new_result.pvalue:.3f}\n")
                    else:
                        print(f"{MODELS_PLOT[m1]} = {MODELS_PLOT[m2]}, real p = {result.pvalue}\n")
            print("-"*20)

    for m in MODELS_PLOT:
        print("="*40)
        for d in [0, 4, 8, 12, 16]:
            newdf = rdf_filtered[rdf_filtered["Model"].eq(m) & rdf_filtered["num_remaining_partial"].eq(d)]
            igo_values = newdf.igo.values
            sed_values = newdf.edit_distance.values

            try:
                result = spstats.wilcoxon(x=igo_values, y=sed_values)
            except ValueError:
                print("wilcoxon failed for ", m, d, ".. skipping")
                continue
            p_value = result.pvalue
            print("Compare IGO and SED for ", m, d, " p value is ", min(1, p_value * 6))
            if 6*p_value < 0.05:
                new_result = spstats.wilcoxon(x=igo_values, y=sed_values, alternative="greater")
                if 6*new_result.pvalue < 0.05:
                    print(f"IGO > SED with p {6*new_result.pvalue:.3f}\n")
                else:
                    print(f"IGO < SED, one sided p {6*new_result.pvalue:.3f}\n")
    print("="*40)


if __name__ == "__main__":
    app.run(main)
