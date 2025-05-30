# cr: tunyuanchang
# image preprocessing
# bbox crop and resize

import os
import json
import argparse
from PIL import Image

ROOT_DIR = f"/media/tunyuan/Backup/Human36M/"
DATA_DIR = "annotations/"
IMAGE_DIR = "images/"

def crop_and_resize(img, img_w, img_h, bbox, target_size=224):
    x_min, y_min, width, height = map(int, bbox)
    side = max(width, height)

    center_x = x_min + width // 2
    center_y = y_min + height // 2

    half_side = side // 2
    new_x_min = max(center_x - half_side, 0)
    new_y_min = max(center_y - half_side, 0)
    new_x_max = new_x_min + side
    new_y_max = new_y_min + side

    if new_x_max > img_w:
        new_x_min = max(img_w - side, 0)
        new_x_max = img_w
    if new_y_max > img_h:
        new_y_min = max(img_h - side, 0)
        new_y_max = img_h

    # Crop and resize using PIL
    cropped = img.crop((new_x_min, new_y_min, new_x_max, new_y_max))
    resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return resized

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "--d", type=str, default='train', help="Output dir")
    parser.add_argument("--subject", "--s", type=int, default=1, help="Subject ID")
    
    args = parser.parse_args()
    subject_id = args.subject
    OUTPUT_DIR = args.dir

    data_path =  os.path.join(ROOT_DIR, DATA_DIR, f'Human36M_subject{subject_id}_data.json')
    with open(data_path, 'r') as f:
                data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    bbox_map = {ann['image_id']: ann['bbox'] for ann in annotations}

    src_path = os.path.join(ROOT_DIR, IMAGE_DIR)
    dst_path = os.path.join(ROOT_DIR, OUTPUT_DIR)
    os.makedirs(dst_path, exist_ok=True)

    for item in images:
        id = item['id']
        w = item['width']
        h = item['height']
        bbox = bbox_map[id]

        src_file = item['file_name']
        dst_file = os.path.basename(src_file)

        src = os.path.join(src_path, src_file)
        dst = os.path.join(dst_path, dst_file)
        try:
            with Image.open(src) as img:
                img = img.convert('RGB')
                img = crop_and_resize(img, w, h, bbox)
                img.save(dst)

        except Exception as e:
            print(f"Skipping {src}: {e}")