import os
import re
import cv2
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def parse_subject(filename):
    # Extract numbers in order: s, act, subact, cam, frame
    match = re.search(r's_(\d+)_act_(\d+)_subact_(\d+)_ca_(\d+)', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Invalid format: {filename}")
    
def parse_image(filename):
    # Extract numbers in order: s, act, subact, cam, frame
    match = re.search(r's_(\d+)_act_(\d+)_subact_(\d+)_ca_(\d+)_(\d+)\.jpg', filename)
    if match:
        return tuple(int(x) for x in match.groups())
    else:
        raise ValueError(f"Invalid format: {filename}")

def parse_joint(filename):
    match = re.match(r"Human36M_subject(\d+)_joint_3d\.json", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid format: {filename}")

def crop_and_resize_human(image, bbox, target_size=224):
    x_min, y_min, x_max, y_max = bbox
    width, height = x_max - x_min, y_max - y_min
    side = max(width, height)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    new_x_min = max(center_x - side // 2, 0)
    new_y_min = max(center_y - side // 2, 0)
    new_x_max = new_x_min + side
    new_y_max = new_y_min + side

    img_w, img_h = image.size
    new_x_max = min(new_x_max, img_w)
    new_y_max = min(new_y_max, img_h)
    new_x_min = max(new_x_max - side, 0)
    new_y_min = max(new_y_max - side, 0)

    human_crop = image.crop((new_x_min, new_y_min, new_x_max, new_y_max))
    resized = TF.resize(human_crop, (target_size, target_size), interpolation=Image.BICUBIC)
    return resized


class PoseDataset(Dataset):
    def __init__(self, image_folder, joint_folder, S=None):
        self.image_folder = image_folder
        self.image_paths = os.listdir(image_folder)

        if S is not None:
            used_subject = []
            for idx, folder in enumerate(self.image_paths):
                s = int(parse_subject(folder))       
                if s in S:
                    used_subject.append(idx)

            self.image_paths[:] = [self.image_paths[i] for i in used_subject]

        # self.resnet_normalize = transforms.Compose([
        #     transforms.ToTensor(),  # Convert PIL image to [C,H,W] tensor
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], 
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])

        # # Load YOLOv7 model
        # self.model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)
        # self.model.eval()
        # self.model.classes = [0]  # detect only 'person'

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        self.joint_data = dict()
        for filename in os.listdir(joint_folder):
            if filename.endswith("joint_3d.json"):
                subject_id = parse_joint(filename)
                if subject_id not in S: continue
                joint_path = os.path.join(joint_folder, filename)
                with open(joint_path, 'r') as f:
                    self.joint_data[subject_id] = json.load(f)

    def __len__(self):
        return len(self.image_paths)
    
    def detect_and_crop(self, img):
        img_rgb = np.array(img)

        # Detect
        results = self.model(img_rgb)
        detections = results.pred[0]

        if len(detections) == 0:
            raise ValueError("No person detected")

        # Use top detection
        x1, y1, x2, y2, conf, cls = detections[0].tolist()
        bbox = [int(x1), int(y1), int(x2), int(y2)]

        # Crop and normalize
        cropped = crop_and_resize_human(img, bbox)
        tensor = self.resnet_normalize(cropped)
        return tensor

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        frame_files = sorted([f for f in os.listdir(image_path)])[:30]

        assert len(frame_files) == 30, f"Expected 30 frames, got {len(frame_files)} in {image_path}"
        
        images = []
        keypoints = []
        for filename in frame_files:
            
            img = Image.open(os.path.join(image_path, filename)).convert('RGB')
            img = self.transform(img)
            images.append(img)

            subject_id, act_id, subact_id, _, frame_id = parse_image(filename)
            keypoint = torch.tensor(self.joint_data[subject_id][str(act_id)][str(subact_id)][str(frame_id)])
            keypoint = keypoint - keypoint[0]
            keypoints.append(keypoint)

        
        video = torch.stack(images, dim=0)  # (30, C, H, W)
        joint = torch.stack(keypoints, dim=0) # (30, J, 3)
        
        return video, joint
