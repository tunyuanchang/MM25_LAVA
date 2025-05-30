import json
import glob
import os

data_folder = f"/media/tunyuan/Backup/Human36M/joints/"

combined_data = {}

id_list = [1,5,6,7,8]

for subject_id in id_list:
    filepath = os.path.join(data_folder, f"Human36M_subject{subject_id}_joint_3d.json")

    with open(filepath, "r") as f:
        data = json.load(f)

    if subject_id not in combined_data:
        combined_data[subject_id] = {}

    for action_id, subactions in data.items():
        action_id = int(action_id)
        if action_id not in combined_data[subject_id]:
            combined_data[subject_id][action_id] = {}

        for subaction_id, frames in subactions.items():
            subaction_id = int(subaction_id)
            if subaction_id not in combined_data[subject_id][action_id]:
                combined_data[subject_id][action_id][subaction_id] = {}

            for frame_id, joints in frames.items():
                frame_id = int(frame_id)+1
                combined_data[subject_id][action_id][subaction_id][frame_id] = joints

                
output_file = f"/media/tunyuan/Backup/Human36M/joint_train/joint_train.json"
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=2)

print(len(combined_data))

