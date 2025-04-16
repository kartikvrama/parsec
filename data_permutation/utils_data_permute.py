"""Helper functions for data generation."""
import sys
from typing import List
from copy import deepcopy
import random
import itertools
import numpy as np

from utils import constants
from utils import data_struct
from utils import utils_data
from utils import utils_eval

NUM_SCENES_PER_PERM = 3


def _update_cache(scene_a, key_a, scene_b, key_b, cache):
    """Update the distance cache."""
    if (key_a, key_b) in cache or (key_b, key_a) in cache:
        return
    try:
        ed = utils_eval.calculate_edit_distance(scene_a, scene_b)[0]
    except ValueError:
        ed = np.inf
    cache[(key_a, key_b)] = ed
    cache[(key_b, key_a)] = ed


def _check_repeating_scenes(delem_list, delem_labels, gscene, glabel, distance_cache, uid):
    # Check that goal scene does not appear in the demonstration sequence
    for elem, perm_label in zip(delem_list, delem_labels):
        assert any(s.objects_on_surface for s in elem["arrangement"])
        _update_cache(
            elem["arrangement"], perm_label, gscene, glabel,
            distance_cache
        )
        if distance_cache[(perm_label, glabel)] == 0:
            print(f"{uid}- Goal scene {glabel} overlaps with {perm_label}.", file=sys.stderr)
            return False
    # Check that the demonstration scenes do not repeat.
    for i, (elem1, perm_label1) in enumerate(zip(delem_list, delem_labels)):
        for j, (elem2, perm_label2) in enumerate(zip(delem_list, delem_labels)):
            if i == j:
                continue
            _update_cache(
                elem1["arrangement"], perm_label1, elem2["arrangement"],
                perm_label2, distance_cache
            )
            if distance_cache[(perm_label1, perm_label2)] == 0:
                print(f"{uid}: Demonstration {perm_label1} overlaps with {perm_label2}.", file=sys.stderr)
                return False
    return True


def permute_example(
    goal_scene, goal_label, demonstration_scenes, demonstration_labels,
    distance_cache, user_id
):
    """Generate demonstration-goal permutations of a given example.

    This function samples 10 permutations (order of removing objects) of each
    scene since sampling all permutations of the goal scene and demonstration
    scenes is not tractable. For each permutation, objects are masked/ moved to
    the final surface one class at a time. The function then generates all
    combinations of partial demonstrations and partial scenes.
    """
    # Permute goal scene.
    # Return None if the goal scene is empty.
    if all(not s.objects_on_surface for s in goal_scene):
        yield None
    partial_scene_array, _ = mask_objects_recursive(
        goal_scene, move_to_unplaced=True, num_permutations=1,
        remove_all=True
    )
    # TODO: this value is wrong! -1 is not necessary.
    num_objects_partial_scene = len(partial_scene_array)-1

    # Permute demonstration scenes.
    demonstration_permutation_array = []
    demonstration_identifier_list = []
    for letter, dscene in zip(
        demonstration_labels, demonstration_scenes
    ):
        id_list = []
        # Skip the demonstration scene if it is empty.
        if all(not s.objects_on_surface for s in dscene):
            print(f"{user_id}- Demonstration scene {letter} is empty, skipping.", file=sys.stderr)
            continue
        dscene_permutations, total_num_objs = mask_objects_recursive(
            dscene, move_to_unplaced=False, num_permutations=1,
            scenes_per_perm=NUM_SCENES_PER_PERM, remove_all=False
        )
        # Store the original demonstration scene.
        dscene_permutations.append(
            {
                "arrangement": dscene, "num_misplaced": 0, "permutation_index": 0,
                "num_remaining": total_num_objs
            }
        )
        # Store permutations with an added demonstration label.
        for elem in dscene_permutations:
            new_elem = deepcopy(elem)
            new_elem["demonstration_label"] = letter
            demonstration_permutation_array.append(new_elem)
            id_list.append(
                f"{letter}-{new_elem['permutation_index']}-{new_elem['num_misplaced']}"
            )
        demonstration_identifier_list.append(id_list)

    # Generate all combinations of permuted demonstrations.
    permutation_product = itertools.product(*demonstration_identifier_list)
    permutation_product_slices = []
    for perm in permutation_product:
        perm = sorted(list(perm))
        if perm not in permutation_product_slices:
            permutation_product_slices.append(perm)
        for l in range(1, len(perm)):
            for comb in itertools.combinations(perm, l):
                comb = sorted(list(comb))
                if comb not in permutation_product_slices:
                    permutation_product_slices.append(comb)
    def _perm_to_demo(perm):
        dlabel, pindex, num_misplaced = perm.split("-")
        return [
            d for d in demonstration_permutation_array if d["demonstration_label"] == dlabel
            and d["permutation_index"] == int(pindex) and d["num_misplaced"] == int(num_misplaced)
        ][0]

    # Combine permuted goal scenes with enumerated demonstration combinations.
    yield len(partial_scene_array)*(1 + len(permutation_product_slices))

    goal_scene_json = utils_data.scene_to_json(goal_scene)
    for partial_elem in partial_scene_array:
        partial_scene_json = utils_data.scene_to_json(partial_elem["arrangement"])
        # Add no demonstration if the partial scene is non-empty.
        result_base = {
            "goal_label": goal_label,
            "goal_scene": goal_scene_json,
            "partial_permutation_index": partial_elem["permutation_index"],
            "num_removed_goal": partial_elem["num_misplaced"],
            "num_remaining_partial": partial_elem["num_remaining"],
            "total_objects_partial": num_objects_partial_scene,
            "partial_scene": partial_scene_json,
        }
        if partial_elem["num_misplaced"] > 0:
            name = (f"{goal_label}_gperm_{partial_elem['permutation_index']}_gnum{partial_elem['num_misplaced']}_dperm_none")
            result = deepcopy(result_base)
            result.update({
                "demonstration_labels": [],
                "demo_permutation_indices": [],
                "num_removed_demonstrations": [],
                "num_demonstrations": 0,
                "demonstration_scenes": [],
                "original_demonstrations": [],
                "num_remaining_demonstrations": 0
            })
            yield name, result
        # Combine partial scene with demonstration permutations.
        for demo_perm_comb in permutation_product_slices:
            demo_elem_list = [_perm_to_demo(d) for d in demo_perm_comb]
            if not _check_repeating_scenes(
                demo_elem_list, demo_perm_comb, goal_scene, goal_label,
                distance_cache, user_id
            ):
                continue
            name = (
                f"{goal_label}_gperm_{partial_elem['permutation_index']}_gnum{partial_elem['num_misplaced']}_dperm{'_'.join(demo_perm_comb)}"
            )
            result = deepcopy(result_base)
            result.update({
                "demonstration_labels": [d["demonstration_label"] for d in demo_elem_list],
                "demo_permutation_indices": [d["permutation_index"] for d in demo_elem_list],
                "num_removed_demonstrations": [d["num_misplaced"] for d in demo_elem_list],
                "num_demonstrations": len(demo_elem_list),
                "demonstration_scenes": [
                    utils_data.scene_to_json(d["arrangement"]) for d in demo_elem_list
                ],
                "original_demonstrations": [
                    utils_data.scene_to_json(
                        demonstration_scenes[demonstration_labels.index(d["demonstration_label"])]
                    ) for d in demo_elem_list
                ],
                "num_remaining_demonstrations": sum(list(
                    d["num_remaining"] for d in demo_elem_list
                ))
            })
            yield name, result


def mask_objects_recursive(
        complete_scene: List[data_struct.SurfaceEntity], move_to_unplaced: bool = True,
        num_permutations: int = 10, scenes_per_perm: int = -1, remove_all: bool = False
    ):
    """Generate all enumerations of partially arranged scenes from a given scene.

    The function also adds an empty surface to the input scene if it does not
    already exist.

        Args:
        complete_scene: List of SurfaceEntity instances.
        move_to_unplaced: Whether to move removed objects to the unplaced surface.
        num_permutations: Number of object orderings to sample.
        scenes_per_perm: Number of scenes per permutation.
        remove_all: Whether to remove all objects from the scene.
    """

    if not complete_scene[-1] == "unplaced_surface":
        complete_scene.append(utils_data.return_unplaced_surface())

    # Deconstruct the goal scene.
    arranged_object_list = []
    object_placement_map = []
    surface_name_list = []
    for s in complete_scene[:-1]:
        arranged_object_list.extend([o.object_id for o in s.objects_on_surface])
        for o in s.objects_on_surface:
            object_placement_map.append((o, s.name))
        surface_name_list.append(s.name)
    arranged_object_list = sorted(arranged_object_list)
    # print("Number of objects: ", len(arranged_object_list))

    # Generate permutations of the object set.
    random.shuffle(arranged_object_list)
    perm_generator = itertools.permutations(arranged_object_list)
    sampled_permutations = []
    for _ in range(num_permutations):
        sampled_permutations.append(next(perm_generator))

    # Generate partially arranged scenes per permutation.
    partial_scene_enumeration_list = []
    for perm_index, permutation in enumerate(sampled_permutations):
        if scenes_per_perm == -1 or scenes_per_perm >= len(permutation) - 1:
            iters_to_save = np.arange(len(permutation) - 1)
        else:
            assert scenes_per_perm > 0
            iters_to_save = np.linspace(
                0, len(permutation)-2, num=scenes_per_perm+1, endpoint=True, dtype=int
            )[1:]  # Skip the first permutation (remove one object).
        # print(f"Creating {scenes_per_perm} scenes out of {len(permutation)-1} possibilities: {[x + 1 for x in iters_to_save]}")
        temp_scene = [deepcopy(s) for s in complete_scene]
        for i, obj_id in enumerate(permutation[:-1]):
            utils_data.remove_object_from_scene(
                temp_scene, obj_id, move_to_unplaced=move_to_unplaced
            )
            # Store every (stride)th partial scene and the final permutation.
            if i in iters_to_save:
                partial_scene_enumeration_list.append({
                    "arrangement": temp_scene,
                    "num_misplaced": i+1,
                    "permutation_index": perm_index,
                    "num_remaining": len(permutation) - i - 1
                })
            temp_scene = [deepcopy(s) for s in temp_scene]
        if remove_all:
            utils_data.remove_object_from_scene(
                temp_scene, permutation[-1], move_to_unplaced=move_to_unplaced
            )
            partial_scene_enumeration_list.append({
                "arrangement": temp_scene,
                "num_misplaced": len(permutation),
                "permutation_index": perm_index,
                "num_remaining": 0
            })
    return partial_scene_enumeration_list, len(sampled_permutations[0])


def object_list_to_binary(object_list, master_list):
    """Get binary matrix from object list."""
    binary_list = []
    for obj in object_list:
        binary_list.append(object_to_binary(obj, master_list))
    return np.stack(binary_list)


def object_to_binary(obj_name, master_list):
    """Get binary vector from object."""
    binary_vec = np.zeros(len(master_list))
    binary_vec[master_list.index(obj_name)] = 1
    return binary_vec
