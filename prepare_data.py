import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_coco_to_yolo(coco_json_path, images_dir, output_labels_dir):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    os.makedirs(output_labels_dir, exist_ok=True)

    #category map
    cat_map = {cat['id']: i for i, cat in enumerate(data['categories'])}

    for ann in tqdm(data['annotations'], desc=f"Converting {os.path.basename(coco_json_path)}"):
        img_id = ann['image_id']
        img_info = images.get(img_id)
        if not img_info:
            continue
            
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)
        bbox = ann['bbox'] # [x, y, w, h]
        x_center = (bbox[0] + bbox[2] / 2) / width
        y_center = (bbox[1] + bbox[3] / 2) / height
        w_norm = bbox[2] / width
        h_norm = bbox[3] / height
        
        class_id = cat_map[ann['category_id']]
        
        label_file = Path(output_labels_dir) / (Path(file_name).stem + ".txt")
        
        with open(label_file, 'a') as lf:
            lf.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

def prepare_dataset():
    base_path = Path("License Plate Recognition.v1-raw-images.coco")
    new_base = Path("dataset_yolo")
    
    for split in ["train", "valid", "test"]:
        print(f"\nProcessing {split}...")
        img_dir = base_path / split
        json_path = img_dir / "_annotations.coco.json"
        
        new_img_dir = new_base / split / "images"
        new_lbl_dir = new_base / split / "labels"
        
        os.makedirs(new_img_dir, exist_ok=True)
        convert_coco_to_yolo(json_path, img_dir, new_lbl_dir)
        for img_file in tqdm(img_dir.glob("*.jpg"), desc=f"Copying {split} images"):
            shutil.copy(img_file, new_img_dir / img_file.name)

if __name__ == "__main__":
    prepare_dataset()
