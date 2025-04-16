from pathlib import Path
import pickle as pkl
import openai
import matplotlib.pyplot as plt
import cv2

# folder = Path("c:/Users/karti/GaTech Dropbox/Kartik Ramachandruni/misc/scene_parsing_results/scenarios/F/observed")
# for file in folder.iterdir():
#     print(file.name)
#     with open(file, "rb") as f:
#         data = pkl.load(f)
#     print(data["parsed"])
#     cv2.imwrite("c:/Users/karti/GaTech Dropbox/Kartik Ramachandruni/misc/scene_parsing_results/scenarios/F/observed_images/" + file.name + ".png", data["image"])
#     print("="*50)

file_path = "c:/Users/karti/GaTech Dropbox/Kartik Ramachandruni/misc/scene_parsing_results/scenarios/F/placements_scenario_f.pkl"
with open(file_path, "rb") as f:
    data = pkl.load(f)
print(data.keys())

rules = data["output_metadata"]["rules"]
print("rules")
for r in rules:
    print(r)
print("="*50)
print(f"Summary: {data['output_metadata']['summary']}")
print("="*50)
placements = data["placements"]
print("placements")
for p in placements:
    print(p)
print("="*50)