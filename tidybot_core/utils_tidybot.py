"""Helper functions for Tidybot."""
from typing import List, Optional
from copy import deepcopy
import random
from utils import constants
from utils import data_struct
from utils import utils_data


def print_colored(text, color='red', end='\n'):
    color_code = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[34m',
    }[color]
    print(f'{color_code}{text}\033[00m', end=end)


def construct_summarization_prompt(objects, receptacles, placements):
    """Constructs a summarization prompt for Tidybot (adapted from source)."""
    summarization_prompt = '''{template}

objects = {objects_str}
receptacles = {receptacles_str}
{placements_str}
# Summary:'''
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    placements_str = '\n'.join(map(lambda x: f'pick_and_place("{x[0]}", "{x[1]}")', placements))
    return summarization_prompt.format(
        template=constants.TEMPLATE_PH, objects_str=objects_str,
        receptacles_str=receptacles_str, placements_str=placements_str
    )


def construct_placement_prompt(
    unplaced_objects, receptacles, summary=None, placed_objects=None, placements=None
):
    """Constructs a placement prompt for Tidybot (adapted from source).
    
    Note that tidybot cannot interpret the placement of pre-arranged objects;
    however, the prompt can be extended to include already placed objects. Args
    to be used for this purpose are `placed_objects` and `placements`.
    """
    if placed_objects is not None or placements is not None:
        raise NotImplementedError("Tidybot cannot interpret pre-arranged objects.")
    placement_prompt = '''{template}

# Summary: {summary}
objects = {objects_str}
receptacles = {receptacles_str}
pick_and_place("{first_object}",'''
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', unplaced_objects)) + ']'
    receptacles_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacles)) + ']'
    prompt = placement_prompt.format(
        template=constants.TEMPLATE_PH, summary=constants.SUMMARY_PH,
        objects_str=objects_str, receptacles_str=receptacles_str,
        first_object=unplaced_objects[0]
    )
    if summary:
        prompt = prompt.replace(constants.SUMMARY_PH, summary)
    return prompt


def scene_to_summary_prompt(
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


def scene_to_placement_prompt(scene: List[data_struct.SurfaceEntity]):
    """Converts a scene struct into a LLM object rearrangement prompt.

    The order of objects on each surface is shuffled to avoid ordering bias.

    Args:
        scene: List of SurfaceEntity instances.

    Returns:
        Object placement prompt.
    """

    unplaced_object_labels = []

    def _serialize_object(name):
        if name in unplaced_object_labels:
            name = f"{name} {unplaced_object_labels.count(name) + 1}"
        return name

    if scene[-1].surface_type != constants.SURFACE_TYPES.NOT_PLACED:
        raise ValueError(f"Last surface is {scene[-1].name} and not 'unplaced_surface'.")
    receptacle_list = list(
        surface.text_label for surface in scene[:-1]
    )

    # Add the unplaced objects.
    unplaced_objects = deepcopy(scene[-1].objects_on_surface)
    random.shuffle(unplaced_objects) # Shuffle to avoid ordering bias.
    for obj in unplaced_objects:
        object_name = _serialize_object(obj.name)
        unplaced_object_labels.append(object_name)

    # Construct the placement prompt w/0 summary.
    placement_prompt = construct_placement_prompt(
        unplaced_objects=unplaced_object_labels, receptacles=receptacle_list
    )
    return placement_prompt, unplaced_object_labels
