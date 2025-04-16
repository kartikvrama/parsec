from typing import Any, List, Dict, Union, Tuple
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
from utils import constants

# Common model configurations
MODELS = [
    "ContextSortLM",
    "TENet",
    "TidyBot-Random",
    "ConSOR",
    "NeatNet",
    "CFFM",
    "CF",
    "APRICOT-NonInteractive",
]

MODELS_PLOT = [
    "CF",
    "TidyBot-Random",
    "ConSOR",
    "APRICOT-NonInteractive",
    "ContextSortLM",
]

# Common color schemes
COLORS = [
    "#fbb4ae",
    "#b3cde3",
    "#ccebc5",
    "#decbe4",
    "#fed9a6",
]

COLORS_DISPARATE = ["#bebada", "#fb8072", "#80b1d3", "#fdb462"]

# Common group definitions
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

GROUP_NAMES = [
    "A:Similar-1D", "B:Similar-2D", "C:Dissimilar",
]

def find_group(env: str, var: int, group_dict: List[Dict] = GROUP_LABEL_DICT) -> str:
    """Find the group label for a given environment and variant.
    
    Args:
        env: Environment name
        var: Variant number
        group_dict: List of group dictionaries to search in
        
    Returns:
        Group label (A, B, C, etc.)
    """
    label_list = list("ABCDEFGHIJKL")[:len(group_dict)]
    for group_id, group in enumerate(group_dict):
        if env in group:
            if var in group[env]:
                return label_list[group_id]
    raise ValueError(f"Could not find group for {env} and {var}.")

def load_fold_dict(fold_path: str) -> Tuple[Dict, int]:
    """Load fold dictionary and extract metadata.
    
    Args:
        fold_path: Path to the PKL file containing data folds
        
    Returns:
        Tuple of (metadata dictionary, number of folds)
    """
    with open(fold_path, "rb") as fp:
        fold_dict_parent = pkl.load(fp)
    num_folds = len(fold_dict_parent) - 1
    metadata = {}
    for fkey, fdf in fold_dict_parent.items():
        if fkey == "metadata":
            continue
        for _, row in fdf["test"].iterrows():
            metadata[row["scene_id"]] = {
                "num_removed_goal": int(row["num_removed_goal"]),
                "num_remaining_partial": int(row["num_remaining_partial"]),
                "user_id": row["scene_id"].split("_", maxsplit=1)[0],
                "group": find_group(row["environment"], row["variant"]),
                "goal_label": row["scene_id"].split("_", maxsplit=2)[1],
            }
    del fold_dict_parent
    return metadata, num_folds

def load_model_results(
    model_folder_dict: Dict[str, str],
    num_folds: int,
    fold_num: int = None
) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict[str, List[str]], List[str]]:
    """Load results from multiple models.
    
    Args:
        model_folder_dict: Dictionary mapping model names to their result folders
        num_folds: Number of folds to load
        fold_num: Optional specific fold number to load
        
    Returns:
        Tuple of (results DataFrame, predictions dictionary, scene IDs per model, model list)
    """
    result_df = None
    predictions_per_model = {model: {} for model in model_folder_dict.keys()}
    
    for model, folder in model_folder_dict.items():
        print(f"Loading results for {model}")
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Results folder {folder} does not exist.")
            
        for fkey, subf in list((f"fold_{i}", folder / f"fold_{i}") for i in range(1, 1 + num_folds)):
            if fold_num is not None and fkey != f"fold_{fold_num}":
                continue
            elif fold_num is not None:
                print(f"ONLY loading {fkey} results for {model}")
                
            if not subf.exists():
                raise FileNotFoundError(f"Cannot load {fkey} results for {model}: {subf} does not exist.")
                
            # Load evaluation results
            df = pd.read_csv(subf / "results.csv")
            df["Model"] = model
            if result_df is None:
                result_df = df.copy()
            else:
                result_df = pd.concat([result_df, df], ignore_index=True)
                
            # Load predictions
            with open(subf / "predictions.pkl", "rb") as fp:
                predictions_per_model[model].update(pkl.load(fp))
                
    # Check if all models (except ConSOR) have the same scene_ids
    model_list = result_df["Model"].unique()
    scene_ids_per_model = result_df.groupby("Model")["scene_id"].unique().to_dict()
    scene_id_sets = [set(scene_ids_per_model[model]) for model in model_list if model != "ConSOR"]
    assert all(scene_id_sets[0] == scene_id_set for scene_id_set in scene_id_sets), "Scene IDs do not match across models."
    
    # Add user_id and goal_label columns
    result_df["user_id"] = result_df["scene_id"].apply(lambda x: x.split("_", maxsplit=1)[0])
    result_df["goal_label"] = result_df["scene_id"].apply(lambda x: x.split("_", maxsplit=2)[1])
    
    return result_df, predictions_per_model, scene_ids_per_model, model_list


def darken_color(hex_color, factor=0.8):
    """
    Darkens a hex color by reducing its brightness.

    Args:
        hex_color (str): The hex color string (e.g., "#bebada").
        factor (float): The factor by which to darken (0 < factor < 1).

    Returns:
        str: The darkened hex color string.
    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Darken each channel
    darkened_rgb = tuple(int(c * factor) for c in rgb)
    
    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(*darkened_rgb)
