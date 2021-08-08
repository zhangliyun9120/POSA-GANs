import numpy as np

import torch

import cv2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# panoptic segmentation:
def get_panoptic_data_img(im_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # img is PIL Image format, so we just use path
    im = cv2.imread(im_path)
    # im = cv2.resize(im, img.size, interpolation=cv2.INTER_AREA)

    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    # object_data map
    object_data = {
        'category_id': [],
        'bbox': [],
        'mask': [],
    }

    object_data_out = v.crop_panoptic_seg_predictions(object_data, panoptic_seg.to("cpu"), segments_info)

    objs, boxes, masks, obj_to_img = output_panoptic_data_img(object_data_out)

    return objs, boxes, masks, obj_to_img


# extract objs, bboxes, maskes :
def output_panoptic_data_img(object_data, image_size=(256, 256)):
    boxes, masks = [], []
    objs = torch.LongTensor(object_data['category_id'])  # int64
    W, H = image_size

    for i, ((x0, y0, x1, y1), mask) in enumerate(zip(object_data['bbox'], object_data['mask'])):
        x0 = x0 / W
        y0 = y0 / H
        x1 = x1 / W
        y1 = y1 / H
        box = [x0, y0, x1, y1]

        mask = mask + 0
        mask = np.int32(mask)

        boxes.append(torch.FloatTensor(box))  # to 32位float类型 tensor
        masks.append(torch.FloatTensor(mask))  # dimension is different so that it cannot convert to tensor format

    # objs = torch.LongTensor(objs)
    boxes = torch.stack(boxes, dim=0)
    masks = torch.stack(masks, dim=0)

    O = objs.size(0)
    obj_to_img = torch.LongTensor(O).fill_(0)

    return objs, boxes, masks, obj_to_img