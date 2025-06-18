import os
import json
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(os.path.dirname(current_dir), "Human36M")
data_folder = os.path.join(root_dir, "annotations")

combined_data = {}
id_list = []

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "--f", type=str, default='train', help="Output filename")
    parser.add_argument("--train", "--t", type=bool, default=True, help="Train/Test")

    args = parser.parse_args()
    filename = args.dir
    if args.train:
        id_list = [1,5,6,7,8]
    else: 
        id_list = [9]

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

                
    output_file = os.path.join(root_dir, filename, f"{filename}.json")
    with open(output_file, "w") as f:
        json.dump(combined_data, f, indent=2)

    print(len(combined_data))

