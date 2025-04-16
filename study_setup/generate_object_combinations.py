"""
Script to generate unique object combinations to instantiate scenes in the
AMT study.
"""

import os
import csv
import random
from typing import Any

from datetime import datetime
import pickle as pkl
from PIL import Image

# CSV fields
IDENTIFIER_FIELD = "fullId"
DATASET_FIELD = "dataset"
NAME_FIELD = "name"
RECEPTACLE_FIELD = "relevant_receptacle"
THUMBNAIL_FIELD = "thumbnail_number"

# Object image dataset folder paths
AMAZON_OBJS_IMG_FOLDER = "/home/kartik/Documents/datasets/amazon_product_images_custom"
SHAPENET_SEM_IMG_FOLDER = (
    "/home/kartik/Documents/datasets/shapenetsem/models-screenshots/screenshots"
)
GOOGLE_OBJS_IMG_FOLDER = "/home/kartik/Documents/datasets/google_objects"

# Default thumbnail number per dataset
THUMBNAIL_DEFAULTS = {"amazon": None, "shapenetsem": 7, "googleobjs": 0}


# Data Config
# TODO: Convert into CLI arguments
CONFIG = {
    "num_problems_per_env": 400,
    "num_objects_per_distr": 10,
    "num_object_categories_per_distr": 6,
    "object_csv_file": "../labels/object_labels_2024_02_29.csv",
    "target_dir": "/home/kartik/Documents/datasets/amt_household_objects_2024-02",
}

def paste_image_and_resize(image: Image, crop=False) -> Image:
    """Pastes an image on a white square background and resizes to 512.

    If crop is True, we use the minimum dimension of the image as the background
    dimension rather than the maximum.

    Args:
        image: PIL image object.
    """
    img_width, img_height = image.size
    if crop:
        background_dim = min(img_width, img_height)
        if img_width > img_height:
            # Remove the left and right parts of the image.
            image = image.crop(
                (
                    (img_width - img_height) // 2,
                    0,
                    (img_width - img_height) // 2 + background_dim,
                    background_dim,
                )
            )
        else:
            # Remove the top and bottom parts of the image.
            image = image.crop(
                (
                    0,
                    (img_height - img_width) // 2,
                    background_dim,
                    (img_height - img_width) // 2 + background_dim,
                )
            )
        img_width, img_height = image.size
    else:
        background_dim = max(img_width, img_height)

    background = Image.new("RGB", [background_dim, background_dim], (255, 255, 255))
    offset = ((background_dim - img_width) // 2, (background_dim - img_height) // 2)
    background.paste(image, offset)
    return background.resize((512, 512))


def return_pkl_object_dict(object_data: dict[str, str], idx: int) -> dict[str, str]:
    """Returns a compressed object dictionary from an object CSV data row.

    Args:
        object_data: CSV dictionary containing the full object data.
        idx: Index of the object in the scene.
    """
    object_pkl_dict = {
        IDENTIFIER_FIELD: f"{object_data[IDENTIFIER_FIELD]}:{idx}",
        NAME_FIELD: f"{object_data[NAME_FIELD]}:{idx}",
    }
    return object_pkl_dict


def generate_object_list_from_set(
    object_id_set: list[str], num_objects_in_scene: int = 10
):
    """Generates a list of objects (scene) from a set of (unique) object.

    Function also returns the index n of an object o_n, where o_n is the n^th
    copy of the unique object id o from object_id_set.

    Args:
        object_id_set: List of unique object identifiers.
        num_objects_in_scene: Total number of objects (including instances)
            in the scene.
    """
    repetition_per_object = [0] * len(object_id_set)
    # Duplicate objects to get [num_objects_per_sample] objects in total.
    while sum(repetition_per_object) != num_objects_in_scene:
        repetition_per_object = random.choices([1, 2], k=len(object_id_set))

    object_id_list = []
    object_indices = []
    for obj_id, repeats in zip(object_id_set, repetition_per_object):
        for index in range(repeats):
            object_id_list.append(obj_id)
            object_indices.append(index)

    return object_id_list, object_indices


def sample_object_combinations(
    num_objects_per_sample: int, object_subset: list[str], object_superset: list[str]
) -> tuple[list[str], list[str]]:
    """Samples equal number of object combinations from within and outside a subset.

    Args:
        num_samples_per_distr: Number of objects per sample. The function will
            sample from both within subset S_sub and outside the subset S - S_sub.
        object_subset: A subset of object categories.
        object_superset: Full set of object categories.
    Returns:
        Equal samples of object combinations from within and outside the subset.
    """

    sample_subset = []
    sample_superset = []

    random.shuffle(object_subset)
    random.shuffle(object_superset)

    for obj in object_superset:
        if len(sample_subset) < num_objects_per_sample:
            if obj in object_subset:
                sample_subset.append(obj)

        if len(sample_superset) < num_objects_per_sample:
            if obj not in object_subset:
                sample_superset.append(obj)

        if len(sample_subset) == len(sample_superset) == num_objects_per_sample:
            return sample_subset, sample_superset

    raise ValueError(
        f"Not enough objects in subset to sample {num_objects_per_sample} combinations."
    )


def _compare_unique_lists(list1: list[Any], list2: list[Any]):
    """Determines if elements of list1 lie in list 2. This only applies to lists
    with unique elements.
    """
    return all(x in list2 for x in list1)


def save_object_images(object_id_to_dict):
    """Save images of objects using the labeled thumbnail index."""
    target_dir = CONFIG["target_dir"]

    for obj_id, obj_data in object_id_to_dict.items():
        obj_dataset = obj_data[DATASET_FIELD]

        if obj_data[THUMBNAIL_FIELD] == "":
            obj_thumbnail = THUMBNAIL_DEFAULTS[obj_dataset]
        else:
            obj_thumbnail = obj_data[THUMBNAIL_FIELD]

        if obj_dataset == "amazon":
            obj_img_path = os.path.join(
                AMAZON_OBJS_IMG_FOLDER,
                f"{obj_id}.jpg",
            )
        elif obj_dataset == "shapenetsem":
            obj_img_path = os.path.join(
                SHAPENET_SEM_IMG_FOLDER, obj_id, f"{obj_id}-{obj_thumbnail}.png"
            )
        elif obj_dataset == "googleobjs":
            obj_img_path = os.path.join(
                GOOGLE_OBJS_IMG_FOLDER, obj_id, f"thumbnails/{obj_thumbnail}.jpg"
            )
        else:
            raise ValueError(f"Unknown dataset {obj_dataset}")

        crop_image = obj_dataset == "googleobjs"
        image = paste_image_and_resize(Image.open(obj_img_path), crop=crop_image)
        image.save(os.path.join(target_dir, f"{obj_id}.png"))


def main():
    random.seed(8514352)
    date_time_stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # Load objects from the CSV file.
    with open(CONFIG["object_csv_file"], "r") as file_csv:
        reader = csv.DictReader(file_csv)
        object_csv = list(reader)
    object_id_to_dict = {d[IDENTIFIER_FIELD]: d for d in object_csv}
    print("Total number of objects", len(object_id_to_dict))

    # Save object images to the target directory.
    os.makedirs(CONFIG["target_dir"], exist_ok=True)
    save_object_images(object_id_to_dict)

    # Categorize objects by their receptacle labels.
    receptacle_to_object_id = {}
    for obj_id in object_id_to_dict:
        obj_recepts = object_id_to_dict[obj_id][RECEPTACLE_FIELD].split(";")
        obj_recepts = [recept.strip() for recept in obj_recepts]  # strip whitespace.
        for recept in obj_recepts:
            if recept in receptacle_to_object_id:
                receptacle_to_object_id[recept].append(obj_id)
            else:
                receptacle_to_object_id[recept] = [obj_id]

    # Each receptacle should have a unique list of objects
    for recept, recept_objects in receptacle_to_object_id.items():
        if any(recept_objects.count(o) > 1 for o in recept_objects):
            raise ValueError(f"Duplicate objects in {recept}")

    print(f"Recepts: {list(r for r in receptacle_to_object_id)}")
    print(
        f"Number of objects per recept: {[len(receptacle_to_object_id[r]) for r in receptacle_to_object_id]}"
    )

    problems_by_receptacle = {recept: [] for recept in receptacle_to_object_id}
    for recept, objects_in_receptacle in receptacle_to_object_id.items():
        object_id_samples_history = {
            "in_recept": [],
            "out_recept": [],
        }  # For checking duplicates.
        print(f"Generating problems for receptacle {recept}")

        max_problems_per_env = (CONFIG["num_problems_per_env"])
        count = 1
        while count <= max_problems_per_env:
            in_recept_sample, out_recept_sample = sample_object_combinations(
                CONFIG["num_object_categories_per_distr"], objects_in_receptacle,
                list(oid for oid in object_id_to_dict),
            )

            # Skip this scene if the object set is repeated.
            if any(
                _compare_unique_lists(in_recept_sample, x)
                for x in object_id_samples_history["in_recept"]
            ):
                continue
            elif any(
                _compare_unique_lists(out_recept_sample, x)
                for x in object_id_samples_history["out_recept"]
            ):
                continue
            else:
                # Add sampled objects to the history of sampled object sets.
                object_id_samples_history["in_recept"].append(in_recept_sample)
                object_id_samples_history["out_recept"].append(out_recept_sample)
                # Make copies of in/out distribution object categories
                # multiple times and add them up.
                (
                    objects_in_recept,
                    object_indexes_in_recept,
                ) = generate_object_list_from_set(
                    in_recept_sample,
                    num_objects_in_scene=CONFIG["num_objects_per_distr"],
                )
                (
                    objects_out_recept,
                    object_indexes_out_recept,
                ) = generate_object_list_from_set(
                    out_recept_sample,
                    num_objects_in_scene=CONFIG["num_objects_per_distr"],
                )
                scene_object_ids = objects_in_recept + objects_out_recept
                scene_object_indices = (
                    object_indexes_in_recept + object_indexes_out_recept
                )
                # Make instances appear together.
                object_shuffling_order = sorted(list(range(len(scene_object_ids))))

                # Save the scene key and object list.
                identifier = f"{date_time_stamp}_{recept}_{count}"
                problems_by_receptacle[recept].append(
                    {
                        "identifier": identifier,
                        "objects": [
                            return_pkl_object_dict(
                                object_id_to_dict[scene_object_ids[x]],
                                scene_object_indices[x],
                            )
                            for x in object_shuffling_order
                        ],
                    }
                )
                count += 1

    # Save object combinations categorized by receptacles in pickle file.
    with open(
        f"problem_sets/object_combinations_{date_time_stamp}.pkl", "wb",
    ) as fpkl:
        pkl.dump(
            {
                "objects_by_recept": receptacle_to_object_id,
                "problems_by_recept": problems_by_receptacle,
            },
            fpkl,
        )


if __name__ == "__main__":
    main()
