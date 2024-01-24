from fastapi import FastAPI, Query
from demo import init_detector, inference_detector
import uvicorn
from fastapi import File, UploadFile
import os.path as osp
import os
import numpy as np
import mmcv
import cv2
import uuid
from fastapi.staticfiles import StaticFiles
from enum import Enum


router = FastAPI() 
model = None

router.mount("/static", StaticFiles(directory="demo_images"), name="/static")

def load_model(config, checkpoint):
    device = "cuda:1"
    # Load model
    model = init_detector(config=config, checkpoint=checkpoint, device=device)
    return model

print("Loading Kfiou: ")
kfiou = load_model(
    "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/train_fliou/r3det_kfiou_ln_swin_tiny_adamw_fpn_1x_dota_ms_rr_oc_2.py", 
    "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/train_fliou/epoch_50.pth"
)


@router.get("/")
async def getRoot():
    return {"message": "Hello World"}

@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    global kfiou
        
    image.filename = f"{uuid.uuid4()}.png"
    with open(f"temp/{image.filename}", "wb") as f:
        f.write(await image.read())
    model = kfiou
    result = inference_detector(model, osp.join("temp", image.filename))
    #show the results
    polygons = []
    for i, bbox in enumerate(result[0]):
        if bbox[-1] < 0.3:
            continue
        xc, yc, w, h, ag = bbox[:5]
        wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
        hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
        p1 = (xc - wx - hx, yc - wy - hy)
        p2 = (xc + wx - hx, yc + wy - hy)
        p3 = (xc + wx + hx, yc + wy + hy)
        p4 = (xc - wx + hx, yc - wy + hy)
        poly = np.int0(np.array([p1, p2, p3, p4]))
        polygons.append((poly))
    img = mmcv.imread(osp.join("temp", image.filename))
    os.remove(osp.join("temp", image.filename))
    for poly in polygons:
        img = cv2.polylines(img, [poly], True, (0, 0, 255), 2)
    result_path = "demo_images/"
    mmcv.imwrite(img, osp.join(result_path, image.filename))
    return {"image_result": osp.join(result_path, image.filename)}

if __name__ == "__main__":
    uvicorn.run("api:router", host="0.0.0.0", port=35000, reload=True)