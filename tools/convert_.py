import os
from tqdm import tqdm
import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    data_path = os.path.join(args.input, "result.json")
    
    with open(data_path, "r") as json_file:
        data = json.load(json_file)
    
    old_name = args.input + "/images"
    new_name = args.input + "/test"
    
    # rename images to test and move output folder
    if os.path.exists(old_name):
        os.system("mv {} {}".format(old_name, new_name))
    if os.path.exists(new_name):
        os.system("mv {} {}".format(new_name, args.output))
    
    id_to_image_name = dict()
    for id, image in enumerate(data['images']):
        image_name = image['file_name'].split('/')[1]
        id_to_image_name[id] = image_name
    
    annotation = dict()
    for ann in data['annotations']:
        image_id = ann['image_id']
        image_name = id_to_image_name[image_id]
        if image_name not in annotation:
            annotation[image_name] = []
        bbox_info = ann['bbox']
        bbox_info = [str(int(bbox)) for bbox in bbox_info]
        result_str = " ".join(bbox_info) + " text 0"
        annotation[image_name].append(result_str)
    
    out_txt_folder = os.path.join(args.output, "ann_test")
    os.makedirs(out_txt_folder, exist_ok=True)
    for image_name in tqdm(annotation):
        out_txt_path = os.path.join(out_txt_folder, image_name.replace(".jpg", ".txt"))
        with open(out_txt_path, "w") as f:
            for bbox_info in annotation[image_name]:
                f.write(bbox_info + "\n")
    
if __name__ == "__main__":
    main()