import os
import re
import json
import torch
from PIL import Image
from collections import defaultdict
from torchvision import transforms
from torch.utils.data import Dataset
    
def parse_image(filename):
    # Extract numbers in order: s, act, subact, cam, frame
    match = re.search(r's_(\d+)_act_(\d+)_subact_(\d+)_ca_(\d+)_(\d+)\.jpg', filename)
    if match:
        return tuple(int(x) for x in match.groups())
    else:
        raise ValueError(f"Invalid format: {filename}")

class PoseDataset(Dataset):
    def __init__(self, root_dir, image_folder, joint_folder, window_size=30, stride=30):
        image_folder = os.path.join(root_dir, image_folder)
        joint_folder = os.path.join(root_dir, joint_folder)

        self.samples = []

        joint_file = os.listdir(joint_folder)[0]
        joint_path = os.path.join(joint_folder, joint_file)
        with open(joint_path, 'r') as f:
                joints = json.load(f)

        image_file = os.listdir(image_folder)

        grouped = defaultdict(list)
        for item in image_file:
            subject_id, action_id, subaction_id, camera_id, frame_id = parse_image(item)
            key = (subject_id, action_id, subaction_id, camera_id)
            grouped[key].append(item)

        for key, images in grouped.items():
            images_sorted = sorted(images)

            for i in range(0, len(images_sorted) - window_size + 1, stride):
                image_seq = images_sorted[i:i + window_size]
                joint_seq = []
                
                for frame in image_seq:
                    subject_id, action_id, subaction_id, camera_id, frame_id = parse_image(frame)

                    joint = torch.tensor(joints[str(subject_id)][str(action_id)][str(subaction_id)][str(frame_id)])
                    joint = joint - joint[0]
                    joint_seq.append(joint)             
                    
                self.samples.append([image_seq, joint_seq])
       
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_seq, joints = self.samples[idx]

        images = []

        for path in image_seq:
            image_path = os.path.join(self.image_folder, path)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            images.append(image)

        images = torch.stack(images)  # Shape: (window_size, C, H, W)
        joints = torch.stack(joints)  # Shape: (window_size, num_joints, 3)

        return images, joints
