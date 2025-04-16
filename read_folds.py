import pickle as pkl
from absl import app
from absl import flags

from utils import utils_data

flags.DEFINE_string("dataset", None, "Path to the dataset folder.")
flags.DEFINE_string("fold", None, "Path to the PKL file containing data folds.")
FLAGS = flags.FLAGS

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    with open(FLAGS.fold, "rb") as fp:
        fold_dict = pkl.load(fp)

    for fkey, fold_df in fold_dict.items():
        if fkey == "metadata":
            continue
        print(f"Fold {fkey}.")

        # Filter data.
        filtered_df_train = utils_data.return_fold_max_observations(
            fold_df["train"],
            fold_df["train"]["user_id"].unique().tolist(),
        )
        filtered_df_val = utils_data.return_fold_max_observations(
            fold_df["val"],
            fold_df["val"]["user_id"].unique().tolist(),
        )
        filtered_df_test = utils_data.return_fold_max_observations(
            fold_df["test"],
            fold_df["test"]["user_id"].unique().tolist(),
        )

        print("Number of training samples:", len(filtered_df_train))
        print("Number of validation samples:", len(filtered_df_val))
        print("Number of test samples:", len(filtered_df_test))
        print("----")


if __name__ == "__main__":
    app.run(main)