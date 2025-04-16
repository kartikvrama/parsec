from typing import List, Tuple
from pathlib import Path
from absl import app, flags
import pickle as pkl
import time
from ast import literal_eval
import openai
from openai import OpenAI
from utils import constants
from utils import utils_data
from utils import utils_eval
from tidybot_core import prompt_template_tidybot
from tidybot_core.tidybot import parse_summary, parse_placements, placement_to_scene
from utils.openai_cache import Completion
from contextsortlm_core import prompt_template_contextsortlm
from contextsortlm_core import utils_contextsortlm

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

def clean_dict_response(
    response_string: str,
    client: OpenAI
):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Return this string in a JSON format and remove duplicate fields."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": response_string
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


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
    return response.choices[0].message.content


def main():
    client = OpenAI()
    completion = Completion(cache_folder=".openai")

    _, surface_constants = utils_data.return_object_surface_constants()

    with open(Path(FLAGS.fold), "rb") as fpkl:
        fold_df_master = pkl.load(fpkl)

    data_folder_master = Path(FLAGS.dataset)
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

        destination_folder_pth = Path(FLAGS.destination)
        destination_folder_pth.mkdir(parents=True, exist_ok=True)

        with open(Path(data_folder_master)/ fkey /"test.pickle", "rb") as fpkl:
            test_data = pkl.load(fpkl)

        for count, scene_id in enumerate(scene_id_list):
            if count % 50 == 0:
                print(f"Processing scene {count}, scene id {scene_id}.")

            if (destination_folder_pth / f"{scene_id}.pkl").exists():
                print(f"Scene {scene_id} already processed, skipping...")
                continue

            scene_dict = test_data[scene_id]

            rule_prompts = [
                utils_contextsortlm.scene_to_rule_prompt(arr).replace(
                    constants.TEMPLATE_PH, prompt_template_tidybot.TIDYBOT_SUMMARY_TEMPLATE
                )
                for arr in scene_dict["observed_arrangement_list"]
            ]
            rules = list(
                completion.create(prompt, model="gpt-4").choices[0].message.content
                for prompt in rule_prompts
            )
            rules = [parse_summary(rule) for rule in rules]

            receptacle_names = list(
                surface_constants[surf.name]["text"]
                for surf in scene_dict["partial"][:-1]
            )

            rule_summarization_prompt = "{'rules':[" + ",".join(rules) + "]" + "," + "'receptacles':[" + ",".join([f"'{name}'" for name in receptacle_names]) + "]}"

            role_prompt_tuple_list = [
                ("system", prompt_template_contextsortlm.CONTEXTSORTLM_SUMMARY_PROMPT[0]),
                ("user", prompt_template_contextsortlm.CONTEXTSORTLM_SUMMARY_PROMPT[1]),
                ("assistant", prompt_template_contextsortlm.CONTEXTSORTLM_SUMMARY_PROMPT[2]),
                ("user", rule_summarization_prompt)
            ]
            response = send_prompt(role_prompt_tuple_list, client)

            try:
                summary_dict = literal_eval(response)
            except SyntaxError as _:
                print("Syntax error, using GPT to clean response.")
                response = clean_dict_response(response, client)
                summary_dict = literal_eval(response)

            if not "summary" in summary_dict:
                raise ValueError(f"Summary not found in response, received: {response}.")
            summary = str(summary_dict["summary"]).strip()

            placement_prompt, unplaced_object_list = utils_contextsortlm.scene_to_placement_prompt(
                scene_dict["partial"], summary=summary
            )
            role_prompt_tuple_list_2 = [
                ("user", prompt_template_contextsortlm.CONTEXTSORTLM_PLACEMENT_PROMPT[0]),
                ("assistant", prompt_template_contextsortlm.CONTEXTSORTLM_PLACEMENT_PROMPT[1]),
                ("user", placement_prompt)
            ]
            placement_response = send_prompt(role_prompt_tuple_list_2, client)
            placements_parsed = parse_placements(
                placement_response, unplaced_object_list
            )

            predicted_scene = placement_to_scene(
                scene_dict["partial"], placements_parsed
            )
            edit_distance, igo, ipc = utils_eval.compute_eval_metrics(
                predicted_scene, scene_dict["goal"]
            )

            rdict = {
                "scene_id": scene_id,
                "partial": scene_dict["partial"],
                "predicted": predicted_scene,
                "rules": rules,
                "summary_prompt": rule_summarization_prompt,
                "summary_response": response,
                "summary": summary,
                "placement_prompt": placement_prompt,
                "placements": placements_parsed,
                "edit_distance": edit_distance,
                "igo": igo,
                "ipc": ipc,
            }
            with open(destination_folder_pth / f"{scene_id}.pkl", "wb") as fpkl:
                pkl.dump(rdict, fpkl)
            
            if count % 100 == 0:
                print(f"Done processing scene {scene_id}.")
    print("Done processing all scenes.")


if __name__ == "__main__":
    main()
