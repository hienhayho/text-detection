# TEXT DETECTION

- This is the code of our group for the final project of `Machine Learning (CS114.O11)` course, based on [`mmrotate`](https://github.com/open-mmlab/mmrotate). 

- This is a text detection project, but you can use this code to train your model with a dataset contains more than 1 classes.

- Because we are researching and using training dataset so I can't upload it here. When we are done with our research, I will public it later.

- `demo_data/` contains our own labeled dataset, which is presented in our report.

- `demo_res/` contains some inferenced images from the model.

## Set up environment

You have 2 options to do that:

**1.** You can refer to [mmrotate installation](https://github.com/open-mmlab/mmrotate?tab=readme-ov-file#installation). (But sometimes it costs much time for setting up from scratch)

**2.** I have set up and saved this `image` to `docker hub` and you can use this instead of setting up enviroment from scratch.

- Require **docker** to run this.

```bash
# Pull the image
docker pull hienhayho/mmrotate

# Start the docker container
docker run -d --gpus all --shm-size=4G -it -v path/to/your/folder:path/to/your/folder --name mmrotate hienhayho/mmrotate:latest bash

# Execute your container
docker exec -it mmrotate bash

# Activate venv
conda activate openmmlab 
```
> Note: If `openmmlab` not exist, please run following code:
```bash
conda env list

conda activate ... ## the env name that exists except (base) env
```

## Prepare dataset
> Note: As mentioned before, you can use this code for diffirent and much many classes dataset. You must only follow the structure below.

    data
    ├── images
    |    ├── 001.jpg
    |    ├── 002.jpg
    |    ├── ... 
    ├── ann_train
    |    ├── 001.txt
    |    ├── 002.txt
    |    ├── ...             
    ├── ann_val                   
    |    ├── 100.txt
    |    ├── 101.txt
    |    ├── ...

- Please aware that `images/`contains all images for train and validation, and the `ann_train/` contains annotations for trainning images, `ann_val/` contains annotations for validation images.

- The format of `.txt` file in `ann_train/` or `ann_val/` is below, while 8 first numbers refer to 4 points of a bounding box clockwisely from top left. Following that is the class name of that bounding box and the difficulty (you can set it to 0).

```
49 56 83 56 83 70 49 70 text 0
...
```

## Training
**1. About the pretrained model, you can download it from**: [here](https://drive.google.com/file/d/1_puVSraP5g7w52JzChNtECsQZ-B_EUi-/view?usp=sharing)

**2. In config file:**  `train/r3det_kfiou_ln_swin_tiny_adamw_fpn_1x_dota_ms_rr_oc_2_2.py`. 
- Please set these values to your dataset:

```python
# Your local dataset path
data_root = "..." # Your data root
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='DOTADataset',
        ann_file=
        '...', ## Fill here
        img_prefix=
        '...', ## Fill here
    ...
    val=dict(
        type='DOTADataset',
        ann_file=
        '...', ## Fill here
        img_prefix=
        '...', ## Fill here
    test=dict(
        type='DOTADataset',
        ann_file=
        '...', ## Fill here
        img_prefix=
        '...', ## Fill here
...

#Set downloaded pretrained model path 
load_from = "..."
...
```
> Note: If your dataset contains more than 1 class. Please go to `mmrotate/datasets/dota.py` and fix this to your real classes:
```python
CLASSES = ('text',) # Classes which your dataset contains.
...
PALETTE = [(0, 0, 255),] # Color of the bounding box of each class.
...
```

> Note: You can use diffirent configs in `configs/` folder for more models.

**3. Training**

Run this script:

```python
CUDA_VISIBLE_DEVICES=0 python3 \
    tools/train.py \
    your_config_file_path \    # set your config file path
    --work-dir train/
```

## Inference
To inference the model, please run this script:

```python
CUDA_VISIBLE_DIVICES=0 python3 
    demo/image_demo_2.py \
    train/your_config_file \  # set your config file
    train/epoch_50.pth \      # set your checkpoint
    --image-folder ... \      # your folder path contains images need to be inferenced
    --out-folder result/
```

> Note: For more information, please refer to `demo/image_demo_2.py`.