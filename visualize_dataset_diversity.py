#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
from pathlib import Path
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import utils_data
from utils import data_struct
from utils import utils_wn
from visualize import core_compare_struct

# Constants
CONTAINER_TYPE = "fridge"
HOUSEHOLD = 3
DATA_FOLDER = "/path/to/user/arrangements"
OBJECT2ID = None
SURF_CONSTANTS = None

def load_eligible_users():
    """Load and filter eligible users from the user list."""
    user_list = "labels/eligible_users.txt"
    with open(user_list, "r") as f:
        all_users = f.read().splitlines()
    all_users = list(u for u in all_users if not u.startswith("#"))
    assert len(all_users) == 75, f"Incorrect number of eligible users: {len(all_users)}."
    acq_users = utils_data.find_eligible_users()
    eligible_users = list(set(all_users) & set(acq_users))
    print(f"Final number of eligible users: {len(eligible_users)}")
    return eligible_users

def filter_arrangements_by_user(data_tuples, eligible_users):
    """Filter arrangements based on user criteria."""
    arrangement_list = list(t[0] for t in data_tuples)
    metadata = list(t[1] for t in data_tuples)

    user_indexes = list(
        i for i, m in enumerate(metadata)
        if m["user_id"] in eligible_users
        and m["container_type"] == CONTAINER_TYPE
        and m["household"] == HOUSEHOLD
    )
    
    user_ids = []
    arrangements_f2 = []
    objects_used = set()
    object_names_used = set()
    
    for i in user_indexes:
        user_ids.extend([metadata[i]["user_id"]]*len(arrangement_list[i]))
        arrangements_f2.extend(arrangement_list[i])
        for arr in arrangement_list[i]:        
            for surf in arr:
                objects_used.update([o.object_id for o in surf.objects_on_surface])
                object_names_used.update([o.name for o in surf.objects_on_surface])
    
    return user_ids, arrangements_f2, objects_used, object_names_used

def sample_arrangements_from_placements(placement_list, sample_scene, num_samples=5):
    """Generate sample arrangements from a list of placements."""
    global OBJECT2ID
    empty_scene = list(deepcopy(surf) for surf in sample_scene)
    for i in range(len(empty_scene)):
        empty_scene[i].objects_on_surface = []
    
    surface_to_obj_placement = {i:[] for i in range(len(empty_scene))}
    for pl in placement_list:
        obj_name, placement_index = pl
        surface_to_obj_placement[placement_index].append(obj_name)

    sampled_arrangements = []
    for i in range(num_samples):
        sampled_scene = deepcopy(empty_scene)
        for pindex, obj_list in surface_to_obj_placement.items():
            if not obj_list:
                continue
            sampled_objects = random.choices(obj_list, k=3)            
            obj_id_list = list(list(
                k for k, v in OBJECT2ID.items()
                if v["name"] == on
            )[0] for on in sampled_objects)
            obj_entity_list = list(data_struct.ObjectEntity(
                object_id=obj_id, name=OBJECT2ID[obj_id]["text"]
            ) for obj_id in obj_id_list)
            sampled_scene[pindex].add_objects(obj_entity_list)
        sampled_arrangements.append(sampled_scene)
    return sampled_arrangements

def create_persona_arrangements(sample_scene):
    """Create arrangements based on different personas."""
    persona_grocery = [
        ["soy sauce", 0], ["ketchup bottle", 0], ["strawberry jam", 0], ["peanut butter", 0],
        ["milk carton", 1], ["heavy cream", 1], ["cheese", 1], ["sour cream", 1], ["butter", 1],
        ["carton of eggs", 1], ["apple", 2], ["banana", 2], ["carrot", 2],
        ["chicken broth", 3], ["loaf of bread", 3], ["tortillas", 3],
        ["orange juice", 4], ["bottle of lemonade", 4], ["can of soda/pop", 4], ["bottled water", 4],
    ]

    persona_class = [
        ["soy sauce", 0], ["ketchup bottle", 0], ["strawberry jam", 0], ["peanut butter", 0],
        ["heavy cream", 0], ["sour cream", 0], ["apple", 1], ["banana", 1], ["carrot", 1],
        ["loaf of bread", 2], ["tortillas", 2], ["chicken broth", 1],
        ["can of soda/pop", 3], ["carton of eggs", 3], ["butter", 4], ["cheese", 4],
        ["milk carton", 4], ["bottle of wine", 5], ["bottled water", 5], ["orange juice", 5],
    ]

    arrangements_grocery = sample_arrangements_from_placements(persona_grocery, sample_scene, 6)
    arrangements_class = sample_arrangements_from_placements(persona_class, sample_scene, 6)
    
    return arrangements_grocery, arrangements_class

def plot_similarity_matrix(sim_matrix, output_file=None):
    """Plot similarity matrix with overlay."""
    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(sim_matrix, ax=ax1[0], cmap="viridis")
    
    overlay_matrix = np.zeros((len(sim_matrix), len(sim_matrix)))
    for i in range(6):
        overlay_matrix[
            i * len(sim_matrix) // 6 : (i + 1) * len(sim_matrix) // 6,
            i * len(sim_matrix) // 6 : (i + 1) * len(sim_matrix) // 6,
        ] = 1
    print(f"Average similarity among artificial arrangements: {sim_matrix[0:6*2, 0:6*2].mean()}")
    print(f"Average similarity among real arrangements: {sim_matrix[6*2:, 6*2:].mean()}")
    print(f"Similarity between artificial and real arrangements: {sim_matrix[0:6*2, 6*2:].mean()}")
    sns.heatmap(overlay_matrix, ax=ax1[1], cmap="viridis")
    
    if output_file:
        fig1.savefig(output_file)
    plt.close()

def main():
    global OBJECT2ID, SURF_CONSTANTS
    # Load data
    eligible_users = load_eligible_users()
    WORDNET_LABELS = utils_wn.load_wordnet_labels()
    OBJECT2ID, SURF_CONSTANTS = utils_data.return_object_surface_constants()
    filtered_arrangement_dict = utils_data.return_filtered_arrangement_labels(DATA_FOLDER)

    json_files = Path(DATA_FOLDER).rglob("*.json")
    data_tuples = list(
        utils_data.load_user_scene_data(
            json_file, OBJECT2ID, SURF_CONSTANTS
        ) for json_file in json_files
    )

    user_ids, arrangements_f2, objects_used, object_names_used = filter_arrangements_by_user(data_tuples, eligible_users)
    
    # Create sample scene
    sample_scene = [utils_data.clone_surface_type(surf) for surf in arrangements_f2[0]]
    
    # Generate persona arrangements
    arrangements_grocery, arrangements_class = create_persona_arrangements(sample_scene)
    new_arrangement_list = arrangements_grocery + arrangements_class + arrangements_f2
    new_user_ids = ["persona_grocery"]*6 + ["persona_class"]*6 + user_ids
    
    Path("figures/sim_matrix").mkdir(parents=True, exist_ok=True)
    # Calculate and plot similarity matrices
    sim_matrix_vanilla = core_compare_struct.return_sim_matrix(
        new_arrangement_list,
        label_dict=WORDNET_LABELS,
        object_id_dict=OBJECT2ID,
        ignore_duplicates=True
    )
    plot_similarity_matrix(sim_matrix_vanilla, "figures/sim_matrix/sim_matrix_vanilla.pdf")
    
    # Load category data and calculate clustered similarity
    with open("labels/obj_to_cat_fridge.csv", "r") as fcsv:
        data = csv.DictReader(fcsv)
        object_to_category_dict = {row["name"]: row["category"] for row in data}
    
    sim_matrix_clustered = core_compare_struct.return_sim_matrix(
        new_arrangement_list,
        label_dict=object_to_category_dict,
        object_id_dict=OBJECT2ID,
        ignore_duplicates=True
    )
    plot_similarity_matrix(sim_matrix_clustered, "figures/sim_matrix/sim_matrix_clustered.pdf")
    
    # Print unique user IDs
    unique_uids = []
    for u in new_user_ids:
        if u not in unique_uids:
            unique_uids.append(u)
    print("Unique user IDs:", unique_uids)

if __name__ == "__main__":
    main()

