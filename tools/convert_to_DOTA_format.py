import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp
import ujson
from PIL import Image

folder_image_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/images"

def HierText2DOTA(data, images_list, des_folder):
    print("Converting HierText to DOTA format")
    os.makedirs(des_folder, exist_ok=True)
    assert len(data) == 2
    for image in tqdm(images_list):
        temp_image = image.split(".")[0] + ".jpg"
        image_path = osp.join(folder_image_path, temp_image)
        _img = cv2.imread(image_path)
        width, height = _img.shape[1], _img.shape[0]
        image_id = image.split("_")[1].split(".")[0]
        check = False
        for image_data in data[0]["annotations"]:
            if image_data["image_id"] == image_id:
                image_name = "HierText_" + image_data["image_id"] + ".txt"
                image_path = osp.join(des_folder, image_name)
                with open(image_path, "w") as f:
                    for para in image_data["paragraphs"]:
                        for line in para["lines"]:
                            for word in line["words"]:
                                polygon = np.array(word["vertices"]).astype(np.int32).reshape(-1, 2)
                                polygon = polygon.reshape((-1, 1, 2))
                                rect = cv2.minAreaRect(polygon)
                                box = cv2.boxPoints(rect)
                                box = np.int0(box)
                                # width_box = np.linalg.norm(box[0] - box[1])
                                # height_box = np.linalg.norm(box[1] - box[2])
                                # area = width_box * height_box
                                # difficult = None
                                # if area < (width/640*32) * (height/480*32):
                                #     difficult = 1
                                # else:
                                #     difficult = 0
                                difficult = 0
                                assert len(box) == 4
                                box_data = " ".join(box.reshape(-1).astype(str)) + " text {}".format(difficult)
                                f.write(box_data + "\n")
                check = True
        if not check:
            for image_data in data[1]["annotations"]:
                if image_data["image_id"] == image_id:
                    image_name = "HierText_" + image_data["image_id"] + ".txt"
                    image_path = osp.join(des_folder, image_name)
                    with open(image_path, "w") as f:
                        for para in image_data["paragraphs"]:
                            for line in para["lines"]:
                                for word in line["words"]:
                                    polygon = np.array(word["vertices"]).astype(np.int32).reshape(-1, 2)
                                    polygon = polygon.reshape((-1, 1, 2))
                                    rect = cv2.minAreaRect(polygon)
                                    box = cv2.boxPoints(rect)
                                    box = np.int0(box)
                                    # width_box = np.linalg.norm(box[0] - box[1])
                                    # height_box = np.linalg.norm(box[1] - box[2])
                                    # area = width_box * height_box
                                    # difficult = None
                                    # if area < (width/640*32) * (height/480*32):
                                    #     difficult = 1
                                    # else:
                                    #     difficult = 0
                                    # assert len(box) == 4
                                    # assert difficult is not None
                                    difficult = 0
                                    box_data = " ".join(box.reshape(-1).astype(str)) + " text {}".format(difficult)
                                    f.write(box_data + "\n")

def TextOCR2DOTA(data, images_list, des_folder):
    print("Converting TextOCR to DOTA format")
    os.makedirs(des_folder, exist_ok=True)
    assert len(data) == 2
    for image in tqdm(images_list):
        temp_image = image.split(".")[0] + ".jpg"
        image_path = osp.join(folder_image_path, temp_image)
        _img = cv2.imread(image_path)
        width, height = _img.shape[1], _img.shape[0]
        image_id = image.split("_")[1].split(".")[0]
        check_train = True
        img_list = []
        for img in data[0]["anns"].keys():
            if img.startswith(image_id):
                img_list.append(img)
        
        if len(img_list) == 0:
            check_train = False
            for img in data[1]["anns"].keys():
                if img.startswith(image_id):
                    img_list.append(img)
     
        ann_file = osp.join(des_folder, image.split(".")[0] + ".txt")
        with open(ann_file, "w") as f:
            if check_train:
                data_list = data[0]["anns"]
            else:
                data_list = data[1]["anns"]
            for img in img_list:
                polygon = np.array(data_list[img]["points"]).astype(np.int32).reshape(-1, 2)
                polygon = polygon.reshape((-1, 1, 2))
                rect = cv2.minAreaRect(polygon)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                width_box = np.linalg.norm(box[0] - box[1])
                height_box = np.linalg.norm(box[1] - box[2])
                area = width_box * height_box
                difficult = None
                if area < (width/640*32) * (height/480*32):
                    difficult = 1
                else:
                    difficult = 0
                assert len(box) == 4
                assert difficult is not None
                box_data = " ".join(box.reshape(-1).astype(str)) + " text {}".format(difficult)
                f.write(box_data + "\n")

def Other2DOTA(images_list, des_folder):
    print("Converting another to DOTA format")
    os.makedirs(des_folder, exist_ok=True)
    ann_folder = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/labels"
    for image in tqdm(images_list):
        temp_image = image.split(".")[0] + ".jpg"
        image_path = osp.join(folder_image_path, temp_image)
        _img = cv2.imread(image_path)
        width, height = _img.shape[1], _img.shape[0]
        ann_file_name = image.split(".")[0] + ".txt"
        ann_file = osp.join(ann_folder, ann_file_name)
        with open(ann_file, "r") as f:
            lines = f.readlines()
        ann_file = osp.join(des_folder, ann_file_name)
        for line in lines:
            line = line.strip().split(",")
            bbox = np.array(line[:8]).astype(np.int32).reshape(-1, 2)
            width_bbox = np.linalg.norm(bbox[0] - bbox[1])
            height_bbox = np.linalg.norm(bbox[1] - bbox[2])
            area = width_bbox * height_bbox
            difficult = None
            if area < (width/640*32) * (height/480*32):
                difficult = 1
            else:
                difficult = 0
            assert len(line) == 8
            line_data = " ".join(line) + " text {}".format(difficult)
            with open(ann_file, "a") as f:
                f.write(line_data + "\n")



def main():
    folder = "train"
    image_folder = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/{}".format(folder)
    image_names = os.listdir(image_folder)

    hier_text = []
    textOCR_list = []
    another_list = []
    for image_name in tqdm(image_names):
        if image_name.startswith("HierText"):
            hier_text.append(image_name)
        elif image_name.startswith("TextOCR"):
            textOCR_list.append(image_name)
        else:
            another_list.append(image_name)
    assert len(hier_text) + len(textOCR_list) + len(another_list) == len(image_names)

    print("Loading HierText data")
    train_hier_text_data_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/data/train.jsonl"
    val_hier_text_data_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/data/validation.jsonl"
    train_hier_text_data = ujson.load(open(train_hier_text_data_path))
    val_hier_text_data = ujson.load(open(val_hier_text_data_path))

    print("Loading TextOCR data")
    train_textOCR_data_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/data/TextOCR_0.1_train.json"
    val_textOCR_data_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/data/TextOCR_0.1_val.json"
    train_textOCR_data = json.load(open(train_textOCR_data_path))
    val_textOCR_data = json.load(open(val_textOCR_data_path))

    des_folder = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/ann_{}/".format(folder)
    HierText2DOTA([train_hier_text_data, val_hier_text_data], hier_text, des_folder)
    TextOCR2DOTA([train_textOCR_data, val_textOCR_data], textOCR_list, des_folder)
    Other2DOTA(another_list, des_folder)

if __name__ == "__main__":
    main()


