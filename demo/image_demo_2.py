# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

# from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import warnings
import cv2
import json

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import os
import os.path as osp
from tqdm import tqdm
from matplotlib.patches import Polygon

import mmrotate  # noqa: F401


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--image-folder', help='Folder contains images to be tested')
    parser.add_argument('--out-folder', help='Folder contains painted images')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    title = ""
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    image_name = os.listdir(args.image_folder)
    for image in tqdm(image_name):
        result = inference_detector(model, osp.join(args.image_folder, image))
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
        
        img = mmcv.imread(osp.join(args.image_folder, image))
        # img_2 = img.copy()
        img_name = osp.join(args.image_folder, image).split('/')[-1]
        print(img_name)
        # img_ann_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/ann_images_test/TextOCR_83b93410192e9815.txt"
        for poly in polygons:
            img = cv2.polylines(img, [poly], True, (0, 0, 255), 2)
        mmcv.imwrite(img, osp.join(args.out_folder, image))
    # read gt
    # with open(img_ann_path, 'r') as f:
    #     lines = f.readlines()
    
    # for line in lines:
    #     line = line.strip().split(' ')
    #     poly = np.array([line[:8]], dtype=np.int32)
    #     poly = poly.reshape((-1, 1, 2))
    #     img_2 = cv2.polylines(img_2, [poly], True, (0, 0, 255), 2)
    # mmcv.imwrite(img_2, args.out_file.replace('.jpg', '_gt.jpg'))
    
def demo():
    path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/NWD/result_data.json"
    with open(path, "r") as f:
        data = json.load(f)
    for key in data['result'].keys():
        data_res = data['result'][key]
        image_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/NWD/demo_data/images/0bd684c1-54.png"
        polygons = []
        for i, bbox in enumerate(data_res[0]):
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
        img = mmcv.imread(image_path)
        # img_2 = img.copy()
        img_name = image_path.split("/")[-1]
        print(img_name)
        # img_ann_path = "/mlcv1/WorkingSpace/Personal/hienht/Dense/mmrotate/DenseText/Dense/ann_images_test/TextOCR_83b93410192e9815.txt"
        for poly in polygons:
            img = cv2.polylines(img, [poly], True, (0, 0, 255), 2)
        mmcv.imwrite(img, f"demo_images/demo-{key}.jpg")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    # demo()
    
    
