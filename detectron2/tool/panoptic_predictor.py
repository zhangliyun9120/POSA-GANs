import numpy as np
import cv2
import matplotlib.pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


input_path = "../input/1001.jpg"
im = cv2.imread(input_path)
# im = cv2.imread("../input/indoor.jpg")
# plt.figure()

'''
CV2.imread后，直接显示，得到下面这种蓝色的图片，与原图差异很大.
CV2的imread默认存储的颜色空间顺序是BGR，与matplot显示用的imshow的颜色顺序RGB正好相反.
myimg=myimg[...,::-1] or myimg=myimg[:, :, ::-1] # ::-1表示的是逆序
'''

cfg = get_cfg()
# cfg.merge_from_file("panoptic_fpn_R_101_3x.yaml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
# /home/liyun/.torch/iopath_cache/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl
# cfg.MODEL.WEIGHTS = "./output/model_final.pth"

predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

# object_data map
object_data = {
    'object_class': [],
    'category_id': [],
    'bbox': [],
    'mask': [],
}

object_data_out = v.crop_panoptic_seg_predictions(object_data, panoptic_seg.to("cpu"), segments_info)
v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

# output_path = "../output/"
# v = v.crop_panoptic_seg_predictions(input_path, output_path, panoptic_seg.to("cpu"), segments_info)

save_path = "output/output.jpg"
v.save(save_path)

# panoptic segmentation result
plt.imshow(v.get_image()[:, :, ::-1])
plt.show()
