import os
import re
import json

id = 1
image_folder = f"/media/tunyuan/Backup/Human36M/images/"
joint_folder = f"/media/tunyuan/Backup/Human36M/annotations/"

image_paths = os.listdir(image_folder)
joint_path = joint_folder + f'Human36M_subject{id}_data.json'

with open(joint_path, 'r') as f:
    joint_data = json.load(f)

def parse(filename):
    # Extract numbers in order: s, act, subact, cam, frame
    match = re.search(r's_(\d+)_act_(\d+)_subact_(\d+)_ca_(\d+)', filename)
    if match:
        return tuple(int(x) for x in match.groups())
    else:
        raise ValueError(f"Invalid format: {filename}")
    
for idx, path in enumerate(image_paths):
    image_path = os.path.join(image_folder, image_paths[idx])
    filename = sorted([f for f in os.listdir(image_path)])
    
    s, act, sub_act, _ = parse(image_paths[idx])

    if s != id: continue

    print(len(joint_data[str(act)][str(sub_act)]), len(filename))

