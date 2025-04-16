from typing import Any, List, Tuple, Dict
import string
from copy import deepcopy
from utils import constants
from utils import data_struct
from apricot_noquery_core import prompt_template_apricot_noquery as prompt_template
import pdb


def _replace_right(source, target, replacement, replacements=None):
    """Sourced from https://stackoverflow.com/a/14496145."""
    return replacement.join(source.rsplit(target, replacements))


def parse_placements(placement_completion, objects):
    """Parse LLM generated object placements (adapted from source)."""
    placements = []
    first_line = True
    for line in placement_completion.strip().split('\n'):
        if len(line) == 0:
            print('Warning: Stopping since newline was encountered')
            break
        placement_args = line.split(',')
        if len(placement_args) != 2:
            print('Warning: Skipping invalid placement')
            continue
        obj, recep = placement_args
        ## Obj: Remove leading parentheses
        if '(' in obj:
            obj = obj.split('(')[1].strip().replace('"', '').replace("'", '')
        else:
            print('Warning: Found possibly invalid placement')
            obj = obj.strip().replace('"', '').replace("'", '')
        ## Recept: Remove trailing parentheses
        recep = _replace_right(recep.strip(), ')', '', 1)
        recep = recep.replace('"', '').replace("'", '')
        placements.append([obj, recep])
    return placements


def _list_to_itemized_prompt(input_list: List[str]) -> str:
    prompt = "\n".join([f"  -'{x}'" for x in input_list])
    return "\n" + prompt


def _scene_to_state_prompt(scene: List[data_struct.SurfaceEntity]) -> str:
    complete_object_list = []
    unplaced_object_list = []
    def _serialize(name):
        name_str = name
        if name_str in complete_object_list:
            name_str = f"{name_str} {complete_object_list.count(name) + 1}"
        complete_object_list.append(name)
        return name_str
    def _wrap_in_quotes(name):
        return f'"{name}"'
    ## Translate the scene to a string.
    state_string_list = list(
        f'"{surf.text_label}": [{",".join([_wrap_in_quotes(_serialize(obj.name)) for obj in surf.objects_on_surface])}]'
        for surf in scene[:-1]
    )
    if scene[-1].objects_on_surface:
        unplaced_object_list = list(
            _serialize(obj.name) for obj in scene[-1].objects_on_surface
        )
    return "{\n" + ",\n".join(state_string_list) +  "\n}", unplaced_object_list


def _partial_scene_to_prompt(initial_state, receptacle_names, preference):
    placement_prompt_template = '''\
Objects: {objects_str}
Locations: {locations_str}
Initial State: {initial_state_str}
Preference: {preference_str}"'''
    initial_state_str, unplaced_objects = _scene_to_state_prompt(initial_state)
    objects_str = '[' + ', '.join(map(lambda x: f'"{x}"', unplaced_objects)) + ']'
    locations_str = '[' + ', '.join(map(lambda x: f'"{x}"', receptacle_names)) + ']'
    placement_prompt_updated = placement_prompt_template.format(
        objects_str=objects_str,
        locations_str=locations_str,
        initial_state_str=initial_state_str,
        preference_str=preference
    )
    return placement_prompt_updated, unplaced_objects


def generate_prompts_apricot(
    scene_dict: Dict[str, Any], surface_constants: Dict[str, Dict[str, str]]
) -> Tuple[str, str]:
    receptacle_names = list(
        surface_constants[surf.name]["text"]
        for surf in scene_dict["partial"][:-1]
    )
    receptacle_name_prompt = _list_to_itemized_prompt(receptacle_names)
    ## Generate the preference generation query.
    preference_generation_query = [
        (
            "system",
            prompt_template.APRICOT_PREFERENCE_PROMPT.replace(
                prompt_template.LOCATIONS_PH, receptacle_name_prompt
            )
        ),
    ]
    state_prompt_template = '''\
Final state of the environment: {state_prompt_str}
'''
    # pdb.set_trace()
    observed_state_prompts = [
        state_prompt_template.format(state_prompt_str=_scene_to_state_prompt(obs)[0])
        for obs in scene_dict["observed_arrangement_list"]
    ]
    preference_generation_query.extend(
        [
            ("user", prompt)
            for prompt in observed_state_prompts
        ]
    )
    ## Generate the task planning prompt.
    task_planning_query = [
        ("system", prompt_template.APRICOT_PLANNING_PROMPT),
    ]
    placement_prompt_updated, unplaced_object_list = _partial_scene_to_prompt(
        scene_dict["partial"], receptacle_names, constants.SUMMARY_PH
    )
    task_planning_query.append(("user", placement_prompt_updated))

    return preference_generation_query, task_planning_query, unplaced_object_list


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
        matching_asgn_surface = matching_asgn[0][1]
        if not matching_asgn_surface in surf_text_to_index:
            print(f"Surface '{matching_asgn_surface}' not found in scene, skipping.")
            continue
        surf_id = surf_text_to_index[matching_asgn_surface]
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