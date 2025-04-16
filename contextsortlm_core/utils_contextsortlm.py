from typing import List, Optional
from copy import deepcopy
import random
from utils import constants
from utils import data_struct
from tidybot_core.utils_tidybot import construct_summarization_prompt


def _replace_right(source, target, replacement, replacements=None):
    """Sourced from https://stackoverflow.com/a/14496145."""
    return replacement.join(source.rsplit(target, replacements))


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


def construct_placement_prompt_partial(
    placed_objects, unplaced_objects, receptacles, placements, summary=None
):
    """Constructs a placement prompt for Tidybot (adapted from source).
    
    This function is modified from the original Tidybot implementation to
    include pre-arranged object placements in the prompt.
    """
    placement_prompt = '''# Summary: {summary}
objects = {objects_str}
receptacles = {receptacles_str}
{placements_str}
pick_and_place("{first_object}",'''
    objects = placed_objects + unplaced_objects
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    placements_str = '\n'.join(map(lambda x: f'pick_and_place("{x[0]}", "{x[1]}")', placements))
    prompt = placement_prompt.format(
        template=constants.TEMPLATE_PH, summary=constants.SUMMARY_PH,
        objects_str=objects_str, receptacles_str=receptacles_str,
        placements_str=placements_str, first_object=unplaced_objects[0]
    )
    if summary:
        prompt = prompt.replace(constants.SUMMARY_PH, summary)
    return prompt

def scene_to_rule_prompt(
    scene: List[data_struct.SurfaceEntity], summary: Optional[str] = None
):
    """Converts a scene struct into a LLM summarization prompt.

    The order of objects on each surface is shuffled to avoid ordering bias.

    Args:
        scene: List of SurfaceEntity instances.
        summary: Optional summary of the scene, if this is an example scene.
            Defaults to None if this is an evaluation scene.

    Returns:
        Summary generation prompt.
    """

    complete_object_list = []
    placed_object_list = []
    receptacle_list = []
    placement_list = []

    def _serialize_object(name):
        if name in complete_object_list:
            name = f"{name} {complete_object_list.count(name) + 1}"
        return name

    if scene[-1].surface_type != constants.SURFACE_TYPES.NOT_PLACED:
        raise ValueError(f"Last surface is {scene[-1].name} and not 'unplaced_surface'.")

    if summary is not None and len(scene[-1].objects_on_surface) > 0:
        raise ValueError(
            "Summary should not be provided for scenes with unplaced objects."
        )
    for surface in scene[:-1]:
        # Update the placed object lists and list of placement tuples.
        temp_objects = deepcopy(surface.objects_on_surface)
        random.shuffle(temp_objects) # Shuffle to avoid ordering bias.
        for obj in temp_objects:
            object_name = _serialize_object(obj.name)
            placement_list.append((object_name, surface.text_label))
            placed_object_list.append(object_name)
            complete_object_list.append(object_name)
        # Update receptacle list.
        receptacle_list.append(surface.text_label)

    # Construct the summarization prompt.
    summary_prompt = construct_summarization_prompt(
        objects=placed_object_list, receptacles=receptacle_list,
        placements=placement_list
    )
    if summary:
        summary_prompt = summary_prompt.replace(constants.SUMMARY_PH, summary)
    return summary_prompt


def scene_to_placement_prompt(
    scene: List[data_struct.SurfaceEntity], summary: Optional[str] = None
):
    """Converts a scene struct into a LLM object rearrangement prompt.

    The order of objects on each surface is shuffled to avoid ordering bias.

    Args:
        scene: List of SurfaceEntity instances.

    Returns:
        Object placement prompt.
    """

    complete_object_list = []
    placed_object_list = []
    receptacle_list = []
    placement_list = []

    def _serialize_object(name):
        if name in complete_object_list:
            name = f"{name} {complete_object_list.count(name) + 1}"
        return name

    if scene[-1].surface_type != constants.SURFACE_TYPES.NOT_PLACED:
        raise ValueError(f"Last surface is {scene[-1].name} and not 'unplaced_surface'.")
    for surface in scene[:-1]:
        # Update the placed object lists and list of placement tuples.
        temp_objects = deepcopy(surface.objects_on_surface)
        random.shuffle(temp_objects) # Shuffle to avoid ordering bias.
        for obj in temp_objects:
            object_name = _serialize_object(obj.name)
            placement_list.append((object_name, surface.text_label))
            placed_object_list.append(object_name)
            complete_object_list.append(object_name)
        # Update receptacle list.
        receptacle_list.append(surface.text_label)

    # Add the unplaced objects.
    unplaced_object_list = []
    for obj in scene[-1].objects_on_surface:
        object_name = _serialize_object(obj.name)
        unplaced_object_list.append(object_name)
        complete_object_list.append(object_name)

    # Construct the placement prompt.
    placement_prompt = construct_placement_prompt_partial(
        placed_objects=placed_object_list, unplaced_objects=unplaced_object_list,
        receptacles=receptacle_list, placements=placement_list, summary=summary
    )
    return placement_prompt, unplaced_object_list
