"""Plot evaluation metrics for different models."""
from pathlib import Path
from datetime import datetime
from absl import app
from absl import flags
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import constants
from utils.utils_plot import (
    MODELS,
    MODELS_PLOT,
    COLORS,
    GROUP_LABEL_DICT,
    find_group,
    load_fold_dict,
    load_model_results,
)

flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
flags.DEFINE_integer(
    "fold_num",
    None,
    "Fold number to load data from. If None, all folds are loaded."
)
flags.DEFINE_string("contextsortlm", None, "Path to the ContextSortLM results.")
flags.DEFINE_string("tidybot_random", None, "Path to the TidyBot-Random results.")
flags.DEFINE_string("neatnet", None, "Path to the TidyBot results.")
flags.DEFINE_string("consor", None, "Path to the ConSOR results.")
flags.DEFINE_string("declutter", None, "Path to the Declutter results.")
flags.DEFINE_string("cf", None, "Path to the CF results.")
flags.DEFINE_string("cffm", None, "Path to the CF results.")
flags.DEFINE_string("apricot", None, "Path to the APRICOT-NonInteractive results.")
flags.DEFINE_bool("plot", False, "Whether to plot the results.")
FLAGS = flags.FLAGS

matplotlib.rcParams.update({
    "font.size": 64,
    "lines.linewidth": 3,
    "axes.labelsize": 64,
    "axes.titlesize": 64,
    "xtick.labelsize": 75,
    "ytick.labelsize": 75,
})
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    date_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    color_dict = dict(zip(MODELS_PLOT, COLORS))

    # Load fold dictionary and metadata
    scene_id_metadata, num_folds = load_fold_dict(FLAGS.fold)
    
    # Create model folder dictionary
    model_folder_dict = {
        "TidyBot-Random": FLAGS.tidybot_random,
        "ConSOR": FLAGS.consor,
        "CF": FLAGS.cf,
        "ContextSortLM": FLAGS.contextsortlm,
        "APRICOT-NonInteractive": FLAGS.apricot,
    }
    
    # Load results from all models
    result_df, predictions_per_model, scene_ids_per_model, model_list = load_model_results(
        model_folder_dict, num_folds, FLAGS.fold_num
    )

    # Add additional columns to the result_df
    result_df["num_remaining_partial"] = result_df["scene_id"].map(
        lambda x: scene_id_metadata[x]["num_remaining_partial"]
    )
    result_df["num_removed_goal"] = result_df["scene_id"].map(
        lambda x: scene_id_metadata[x]["num_removed_goal"]
    )
    result_df["num_total_goal"] = result_df["num_remaining_partial"] + result_df["num_removed_goal"]
    result_df["placement_acc"] = (result_df["num_removed_goal"] - result_df["edit_distance"]) / result_df["num_removed_goal"]
    assert result_df.placement_acc.min() >= 0, f"Negative placement accuracy found: {result_df.placement_acc.min()}"
    result_df["group"] = result_df["scene_id"].map(
        lambda x: scene_id_metadata[x]["group"]
    )

    # Print placement accuracy per group
    unique_groups = sorted(list(result_df["group"].unique()))
    for model in MODELS:
        avg_placement_acc_aggregate = result_df[result_df.Model.eq(model)]["placement_acc"].mean()
        print(f"{model}: {avg_placement_acc_aggregate:.2f} overall placement accuracy")
        for group in unique_groups:
            aggregate_df_model_env = result_df[
                result_df.Model.eq(model)
                & result_df.group.eq(group)
            ]
            if aggregate_df_model_env.empty:
                continue
            avg_placement_acc = aggregate_df_model_env["placement_acc"].mean()
            print(f"{model} and group {group}: {avg_placement_acc:.2f} avg placement accuracy")
        print("---")
    print("***")

    if not FLAGS.plot:  # Skip plotting if False.
        print("Skipping plotting.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(75, 25))
    fig2, ax2 = plt.subplots(1, figsize=(40, 25))
    metric_labels = {
        "edit_distance": "SED",
        "igo": "IGO",
    }
    count = 0
    # Sort result_df in specific order of models
    result_df = result_df[result_df.Model.isin(color_dict.keys())]
    result_df["Model"] = pd.Categorical(result_df["Model"], MODELS_PLOT)
    
    # ---- Plotting ----
    plotting_interval = 4
    max_x = 15

    rdf_filtered = result_df[
        result_df["num_remaining_partial"].lt(max_x)
        & result_df["num_remaining_partial"].apply(lambda x: x % plotting_interval == 0)
    ]
    rdf_filtered = rdf_filtered.sort_values("num_remaining_partial")
    rdf_filtered.num_remaining_partial = rdf_filtered.num_remaining_partial.astype(str)
    
    ylim = 14  # rdf_filtered["edit_distance"].max()
    for i, (metric, metric_label) in enumerate(metric_labels.items()):
        add_legend = i == 1
        sns.boxplot(
            data=rdf_filtered,
            x="num_remaining_partial",
            y=metric,
            hue="Model",
            ax=ax[i],
            legend=add_legend,
            palette=color_dict,
            width=0.8,
            linewidth=3,
            showfliers=False,
            showmeans=True,
            meanprops={'marker':'+','markerfacecolor':'white','markeredgecolor':'black','markersize':'24', 'markeredgewidth': 3},
        )
        ax[i].set_yticks(np.arange(0, ylim+1, 2))
        if add_legend:
            sns.move_legend(
                ax[i],
                loc="upper left",
                bbox_to_anchor=(1, 1),
                ncol=1,
                title=None
            )
        ax[i].set(
            ylim=(0, ylim + 1),
            xlabel="Number of objects in S_P",
            ylabel=metric_label,
        )
        count += 1

    # Add bar plot for placement accuracy by environment group
    # Sort the data by group alphabetically and model in specific order
    result_df_sorted = result_df.sort_values('group')
    result_df_sorted['Model'] = pd.Categorical(result_df_sorted['Model'], MODELS_PLOT)
    result_df_sorted = result_df_sorted.sort_values(['group', 'Model'])
    
    sns.barplot(
        data=result_df_sorted,
        x="group",
        y="placement_acc",
        hue="Model",
        ax=ax2,
        palette=color_dict,
        errorbar=None,
        capsize=0.1,
    )
    ax2.set(
        xlabel="Environment Category",
        ylabel="Placement Accuracy",
        ylim=(0, 0.7),
    )
    ax2.tick_params(axis='x', rotation=45)
    sns.move_legend(
        ax2,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        ncol=1,
        title=None
    )

    fig.tight_layout()
    fig.savefig(f"figures/{date_time_str}_perror.png", format="png")
    fig2.tight_layout()
    fig2.savefig(f"figures/{date_time_str}_pacc.png", format="png")


if __name__ == "__main__":
    app.run(main)
