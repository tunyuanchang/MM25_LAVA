import os
import re
import json

id = 1
joint_folder = f"/media/tunyuan/Backup/Human36M/annotations/"
joint_path = joint_folder + f'Human36M_subject{id}_data.json'

with open(joint_path, 'r') as f:
    joint_data = json.load(f)


print(joint_data)
for act in joint_data:

    for subact in joint_data[str(act)]:
        
        for frame in joint_data[str(act)][str(subact)]:
            continue
