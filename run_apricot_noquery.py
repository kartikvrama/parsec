import os
from typing import List, Tuple
from ast import literal_eval
from pathlib import Path
from absl import app, flags
import pickle as pkl
import time
import openai
from openai import OpenAI
from utils import constants
from utils import utils_data
from utils import utils_eval
from utils.openai_cache import Completion
from apricot_noquery_core import utils_apricot_noquery

flags.DEFINE_string(
    "fold",
    None,
    "Path to folder containing evaluation prompts.",
)
flags.DEFINE_string(
    "dataset",
    None,
    "Path to folder containing evaluation prompts.",
)
flags.DEFINE_string(
    "destination_folder",
    None,
    "Path to destination folder for tidybot responses."
)
FLAGS = flags.FLAGS

def send_prompt(
    role_prompt_tuple_list: List[Tuple[str, str]],
    client: OpenAI
):
    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                    for role, prompt in role_prompt_tuple_list
                ],
                temperature=1,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "text"},
            )
            success = True
        except (openai.RateLimitError, openai.InternalServerError) as e:
            print(f"Open ai error: {e}, sleeping for 0.2 seconds...")
            time.sleep(60 / 300)
    cost =  response.usage.prompt_tokens*35*1e-6 + response.usage.completion_tokens*60*1e-6
    return response.choices[0].message.content, cost


def _process_preference_json(response: str) -> str:
    try:
        response_cleaned = literal_eval(response)
    except SyntaxError as e:
        if "\n" in response:
            response = response.replace("\n", "")    
        if response.startswith("```json") or response.endswith("```"):
            response = response[response.index("{"):response.rindex("}")+1]
        if "'" in response:
            response = response.replace("'", "\"")
        response_cleaned = literal_eval(response)
    return response_cleaned


def main():
    client = OpenAI()
    completion = Completion(cache_folder=".openai")

    fold_folder = "/coc/flash5/kvr6/data/declutter_user_data/folds-2024-07-20/out_distribution.pkl"
    data_folder_master = "/coc/flash5/kvr6/data/declutter_user_data/batch-2024-07-20/out_distribution"
    destination_folder = "/coc/flash5/kvr6/dev/robo_declutter/logs/apricot_noquery_responses/"
    _, surface_constants = utils_data.return_object_surface_constants()

    with open(Path(fold_folder), "rb") as fpkl:
        fold_df_master = pkl.load(fpkl)

    data_folder_master = Path(data_folder_master)
    for fkey, fold_df_dict in fold_df_master.items():
        if fkey == "metadata":
            continue
        print(f"Processing fold: {fkey}")

        fold_df_test = fold_df_dict["test"]
        fold_df_test["goal_label"] = fold_df_test.scene_id.apply(
            lambda s: s.split("_", maxsplit=2)[1]
        )
        fold_df_test = utils_data.return_fold_max_observations(
            fold_df_test,
            fold_df_test.user_id.unique().tolist(),
        )

        scene_id_list = fold_df_test.scene_id.to_list()
        print(f"Number of scenes: {len(scene_id_list)}")

        destination_folder_pth = Path(destination_folder)
        destination_folder_pth.mkdir(parents=True, exist_ok=True)

        with open(Path(data_folder_master)/ fkey /"test.pickle", "rb") as fpkl:
            test_data = pkl.load(fpkl)

        for count, scene_id in enumerate(scene_id_list):
            if (destination_folder_pth / f"{scene_id}.pkl").exists():
                print(f"Scene {scene_id} already processed, skipping...")
                continue

            scene_dict = test_data[scene_id]
            preference_generation_query, task_planning_query, unplaced_object_list = utils_apricot_noquery.generate_prompts_apricot(
                scene_dict, surface_constants
            )
            preference_response, cost_pref = send_prompt(preference_generation_query, client)
            try:
                preference_response_cleaned = _process_preference_json(preference_response)
                preference = preference_response_cleaned["preference"]
            except:
                print(f"Raw response: {preference_response}")
                print(f"Error in cleaning preference response for scene {scene_id}, skipping.")
                continue
            new_query = task_planning_query[1][1].replace(constants.SUMMARY_PH, preference)
            task_planning_query_updated = (
                task_planning_query[0],
                (task_planning_query[1][0], new_query)
            )
            task_planning_response, cost_plan = send_prompt(task_planning_query_updated, client)

            ## Parse the placements and generate predicted scene.
            placement_response = task_planning_response[task_planning_response.find("pick_and_place"):]
            placements_parsed = utils_apricot_noquery.parse_placements(
                placement_response, unplaced_object_list
            )
            predicted_scene = utils_apricot_noquery.placement_to_scene(
                scene_dict["partial"], placements_parsed
            )
            edit_distance, igo, ipc = utils_eval.compute_eval_metrics(
                predicted_scene, scene_dict["goal"]
            )

            rdict = {
                "scene_id": scene_id,
                "cost": cost_pref + cost_plan,
                "partial": scene_dict["partial"],
                "predicted": predicted_scene,
                "preference_query": preference_generation_query,
                "preference_response": preference_response,
                "preference": preference,
                "task_planning_query": task_planning_query,
                "task_planning_response": task_planning_response,
                "placements": placements_parsed,
                "edit_distance": edit_distance,
                "igo": igo,
                "ipc": ipc,
            }
            with open(destination_folder_pth / f"{scene_id}.pkl", "wb") as fpkl:
                pkl.dump(rdict, fpkl)

            print(f"Done processing scene {scene_id}.")
    print("Done processing all scenes.")


if __name__ == "__main__":
    main()
