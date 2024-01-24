import os.path as osp
import os
from tqdm import tqdm
import torch
import numpy as np
from tabulate import tabulate
import cv2
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

def dist_torch(point1, point2):
    """Calculate the distance between two points.

    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).

    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)

def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes

def main():
    path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/ann_images_test"
    img_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/images_test"
    data = {}
    proposal_num = np.arange(5, 95, 5)
    for num in proposal_num:
        data[num] = [0, 0, 0, 0]
    for file_name in tqdm(os.listdir(path)):
        img_name = os.path.join(img_path, file_name.split(".")[0] + ".png")
        img = cv2.imread(img_name)
        height, width, _ = img.shape
        img_area = height*width
        
        with open(os.path.join(path, file_name), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                bbox = [int(x) for x in line[:8]]
                rbbox = poly2obb_oc(torch.tensor(bbox, dtype = torch.float32))
                theta = rbbox[0][-1]
                for num in proposal_num:
                    if theta >= (np.pi*num)/180:
                        data[num][0] += 1
                        poly = np.array(bbox).reshape(-1, 2)
                        poly = torch.from_numpy(poly)
                        area = Polygon(poly).area
                        if area/img_area <= (32*32)/(640*480):
                            data[num][1] += 1
                        elif area/img_area <= (96*96)/(640*480):
                            data[num][2] += 1
                        else:
                            data[num][3] += 1
    
    for num in proposal_num:
        if data[num][0]:
            data[num].append(np.round(data[num][1]/data[num][0], 3))
            data[num].append(np.round(data[num][2]/data[num][0], 3))
            data[num].append(np.round(data[num][3]/data[num][0], 3))
        else:
            data[num].append(0)
            data[num].append(0)
            data[num].append(0)

    table = []
    for num in proposal_num:
        table.append([num, data[num][0], data[num][4], data[num][5], data[num][6]])
    print(tabulate(table, headers=["Angle", "Total", "Small", "Medium", "Large"]))
                
def visualize(path="/mlcv1/WorkingSpace/Personal/hienht/Dense/NWD/DenseText/Dense/images/HierText_0a362efe964bc6fc.jpg"):
    file_label_name = path.split("/")[-1].replace(".jpg", ".txt")
    label_fol = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/ann_images_test"
    label_file_path = osp.join(label_fol, file_label_name)
    image = Image.open(path)
    draw = ImageDraw.Draw(image)
    with open(label_file_path, "r") as f:
        data = f.readlines()
    for line in data:
        line_data = line.split(" ")
        bbox = [int(x) for x in line_data[:8]]
        # bbox = np.array(bbox).reshape(-1, 2)
        draw.polygon(bbox, outline='red', width=2)
    image.save("demo.jpg")
      
if __name__ == '__main__':
    visualize()