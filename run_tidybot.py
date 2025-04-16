"""Script to generate tidybot responses from pre-generated prompts using GPT-4."""
import json
from pathlib import Path
from absl import app, flags
from tidybot_core import tidybot

flags.DEFINE_string(
    "prompt_directory",
    None,
    "Path to folder containing evaluation prompts.",
)
flags.DEFINE_string(
    "destination_folder",
    None,
    "Path to destination folder for tidybot responses."
)
FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    model = "gpt-4"
    print(f"Using model: {model}")

    if FLAGS.prompt_directory is None:
        raise ValueError("Please provide a prompt folder.")
    prompt_directory = Path(FLAGS.prompt_directory)

    destination_folder = Path(FLAGS.destination_folder)
    if not destination_folder.exists():
        raise ValueError(f"Destination folder <{destination_folder}> does not exist.")

    # Load the prompts per fold.
    folds = list(f for f in prompt_directory.glob('**/*') if f.is_dir())
    for folder in folds:
        print(f"Processing fold: {folder.name}")
        if not folder.name.startswith("fold"):
            raise ValueError(f"Folder <{folder.name}> does not start with 'fold'.")
        file_list = folder.glob("*.json")
        prompt_dict_list = []
        for ct, fname in enumerate(file_list):
            if fname.name == "metadata.json":
                continue
            if ct % 250 == 0:
                print(f"Completed {ct} scenes.")
            with open(folder / fname.name, "r") as fjson:
                prompt_dict = json.load(fjson)
            if (destination_folder / f"{prompt_dict['key']}_answer.json").exists():
                print(f"Response for {prompt_dict['key']} exists, skipping...")
                continue
            prompt_dict_list.append(prompt_dict)

        # Generate and save responses. (Turn on verbose for debugging.)
        response_generator = tidybot.generate_responses(
            prompt_dict_list, model=model, verbose=False
        )
        for count, response in enumerate(response_generator):
            if count % 50 == 0:
                print(f"Generated {count} / {len(prompt_dict_list)} responses - currently at {response['key']}.")
            with open(destination_folder / f"{response['key']}_answer.json", "w") as fjson:
                json.dump(response, fjson, indent=2)


if __name__ == "__main__":
    app.run(main)
