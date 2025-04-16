"""Functions to run and evaluate tidybot."""
import string
from copy import deepcopy

from utils import constants, openai_cache
from tidybot_core import utils_tidybot


def _replace_right(source, target, replacement, replacements=None):
    """Sourced from https://stackoverflow.com/a/14496145."""
    return replacement.join(source.rsplit(target, replacements))


def placement_to_scene(partial_scene, placement_list):
    """Convert object placements into a predicted scene configuration."""

    surf_text_to_index = {}
    for i, surf in enumerate(partial_scene[:-1]):
        surf_text_to_index[surf.text_label] = i

    new_scene = list(deepcopy(s) for s in partial_scene)
    list_unplaced_objects = list(obj for obj in new_scene[-1].objects_on_surface)
    temp_placements = []
    for i, (obj, surf) in enumerate(placement_list):
        if obj[-1].isdigit():
            obj = obj.rstrip(string.digits).strip()
        if "," in obj:
            print(f"Object name '{obj}' contains a comma, re-splitting.")
            obj = obj.split(",")[0].strip()
        if "," in surf:
            print(f"Surface name '{surf}' contains a comma, re-splitting.")
            surf = surf.split(",")[1].strip()
        temp_placements.append((obj, surf))

    for obj in list_unplaced_objects:
        # Find the matching placement assignment for this object.
        matching_asgn = [
            tup for tup in temp_placements if tup[0] == obj.name
        ]
        if not matching_asgn:
            print(f"No matching placement for object {obj.name}, skipping.")
            continue
        if not matching_asgn[0][1] in surf_text_to_index:
            print(f"Surface '{matching_asgn[0][1]}' not found in scene, skipping.")
            continue
        surf_id = surf_text_to_index[matching_asgn[0][1]]

        # Update new scene.
        new_scene[-1].objects_on_surface.remove(obj)
        new_scene[surf_id].add_objects([obj])
        
        # Remove matching assignment from placements.
        temp_placements.remove(matching_asgn[0])

    if temp_placements:
        print(f"Following placements did not have matching unplaced objects: {temp_placements}.")

    if new_scene[-1].objects_on_surface:
        print(f"Scene still has unplaced objects: {list(o.name for o in new_scene[-1].objects_on_surface)}")

    return new_scene


def find_name_mismatches(partial_scene, placement_list):
    """Find object-location tuples that don't match due to naming differences."""
    mismatches_obj = []
    mismatches_surf = []
    mismatches_obj_surf = []

    # Create lookup dictionaries for scene objects and surfaces
    scene_objects = {obj.name for obj in partial_scene[-1].objects_on_surface}
    scene_surfaces = {surf.text_label for surf in partial_scene[:-1]}
    
    # Process placement list similar to placement_to_scene
    temp_placements = []
    for obj, surf in placement_list:
        # Apply the same normalization as in placement_to_scene
        if obj[-1].isdigit():
            obj = obj.rstrip(string.digits).strip()
        if "," in obj:
            obj = obj.split(",")[0].strip()
        if "," in surf:
            surf = surf.split(",")[1].strip()
        temp_placements.append((obj, surf))
    
    for placement_obj, placement_surf in temp_placements:
        if placement_obj in scene_objects and placement_surf in scene_surfaces:
            continue
        else:
            if placement_obj in scene_objects and placement_surf not in scene_surfaces: # surface mismatch
                mismatches_surf.append((placement_obj, placement_surf))
            if placement_surf in scene_surfaces and placement_obj not in scene_objects: # object mismatch
                mismatches_obj.append((placement_obj, placement_surf))
            if placement_obj not in scene_objects and placement_surf not in scene_surfaces:
                mismatches_obj_surf.append((placement_obj, placement_surf))
    return mismatches_obj, mismatches_surf, mismatches_obj_surf


def generate_responses(prompt_dict_list, model="gpt-4-turbo-preview", verbose=False):
    """Evaluate a pair of prompts and return the processed responses."""
    completion = openai_cache.Completion(cache_folder=".openai")
    for prompt_dict in prompt_dict_list:
        summary_response = completion.create(
            prompt_dict["summary_prompt"], model=model
        )
        summary_completion = summary_response.choices[0].message.content
        if verbose:
            print(prompt_dict["summary_prompt"], end='')
            utils_tidybot.print_colored(summary_completion, 'blue')
            print('\n' + 10 * '-' + '\n')
            print(f"Used {summary_response.usage.total_tokens} tokens")

        summary = parse_summary(summary_completion)
        placement_prompt = prompt_dict["placement_prompt"].replace(
            constants.SUMMARY_PH, summary
        )
        placement_response = completion.create(placement_prompt, model=model)
        placement_completion = placement_response.choices[0].message.content
        if verbose:
            print("Unplaced objects:", prompt_dict["unplaced_objects"])
            print(placement_prompt, end='')
            utils_tidybot.print_colored(placement_completion, 'blue')
            print('\n' + 10 * '-' + '\n')
            print(f"Used {placement_response.usage.total_tokens} tokens")

        placements = parse_placements(
            placement_completion, objects=prompt_dict["unplaced_objects"]
        )
        if verbose:
            for (o, r) in placements:
                print(f"{o} -> {r}")
        yield {
            "key": prompt_dict["key"],
            "summary_completion": summary_completion,
            "summary": summary,
            "placement_completion": placement_completion,
            "placements": placements,
            "summary_tokens": summary_response.usage.total_tokens,
            "placement_tokens": placement_response.usage.total_tokens,
        }


def parse_summary(summarization_completion):
    """Parse LLM generated summary (adapted from source)."""
    lines = [l for l in map(str.strip, summarization_completion.split('\n')) if len(l) > 0]
    if len(lines) > 1:
        print('Warning: Using first line of multi-line summary')
    return lines[0]


def parse_placements(placement_completion, objects):
    """Parse LLM generated object placements (adapted from source)."""
    placements = []
    first_line = True
    for line in placement_completion.strip().split('\n'):
        if first_line:
            obj = objects[0]
            recep = line
            first_line = False
        else:
            if len(line) == 0:
                print('Warning: Stopping since newline was encountered')
                break
            placement_args = line.split(',')
            if len(placement_args) != 2:
                print('Warning: Skipping invalid placement')
                continue
            obj, recep = placement_args
            if '(' in obj:
                obj = obj.split('(')[1].strip().replace('"', '')
            else:
                print('Warning: Found possibly invalid placement')
                obj = obj.strip().replace('"', '')
        recep = _replace_right(recep.strip(), ')', '', 1)
        recep = recep.replace('"', '')
        placements.append([obj, recep])
    return placements
