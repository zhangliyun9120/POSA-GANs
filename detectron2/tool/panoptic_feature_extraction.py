
import json
import os
import random
import torch

import cv2
import numpy as np
from matplotlib import pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image


NUM_OBJECTS = 36


def doit(raw_image):
    with torch.no_grad():
        # 1, Get W and H of raw image input
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))

        # 2, Preprocessing raw image
        image = predictor.aug.get_transform(raw_image).apply_image(raw_image)  # transform the W and H of raw image
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)  # get a ImageList of Detectron2

        # 3, Get feature of backbone FPN
        features = predictor.model.backbone(images.tensor)
        print("Backbone Features:", features)  # a dict for backbone feature (p2, p3, p4, p5, p6) layers

        # 4, Get proposal feature of proposal_generator RPN
        proposals, _ = predictor.model.proposal_generator(images, features)  # a list proposed boxes for image regions
        print("Generate proposals with RPN:", proposals)

        proposal = proposals[0]
        print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # 5, Get instances of roi_heads
        instances, _ = predictor.model.roi_heads(images, features, proposals)  # a list instances [16, 1, 28, 28] object

        # 6, Get mask feature of mask_pooler after ROIAlign
        mask_features = [features[f] for f in predictor.model.roi_heads.in_features]  # a Tensor mask [16, 256, 14, 14]
        mask_features = predictor.model.roi_heads.mask_pooler(mask_features, [x.pred_boxes for x in instances])

        # mask_predictions = predictor.model.roi_heads.mask_head(mask_features)

        # 7, Get box feature of box_pooler after ROIAlign
        box_features = [features[f] for f in predictor.model.roi_heads.in_features]  # a Tensor mask [16, 256, 7, 7]
        box_features = predictor.model.roi_heads.box_pooler(box_features, [x.pred_boxes for x in instances])

        box_features = predictor.model.roi_heads.box_head(box_features)
        box_predictions = predictor.model.roi_heads.box_predictor(box_features)

        # 8, Get stuff semantic feature
        sem_seg = predictor.model.sem_seg_head(features)
        print("Generate sem_seg with FPN:", sem_seg[0].shape)  # torch.Size([1, 54, 800, 1024])

        return sem_seg[0]

        # get proposed boxes + rois + features + predictions from RPN proposal boxes directly
        # proposal_boxes = [x.proposal_boxes for x in proposals]   # a Tensor boxes [300, 4]
        # box_features = predictor.model.roi_heads.box_pooler(    # a Tensor mask [300, 256, 7, 7]
        #     [features[f] for f in predictor.model.roi_heads.in_features], proposal_boxes)

        # WE CAN USE THE PREDICTION CLS TO FIND TOP SCOREING PROPOSAL BOXES!
        # pred_instances, losses = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        # proposal_boxes = [x.proposal_boxes for x in proposals]
        # features = [features[f] for f in predictor.model.roi_heads.in_features]
        # box_features = predictor.model.roi_heads._shared_roi_transform(
        #     features, proposal_boxes
        # )

        # print("Box proposals features:", box_features)
        # feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)
        #
        # # Predict classes and boxes for each proposal.
        # pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        # outputs = FastRCNNOutputs(
        #     predictor.model.roi_heads.box2box_transform,
        #     pred_class_logits,
        #     pred_proposal_deltas,
        #     proposals,
        #     predictor.model.roi_heads.smooth_l1_beta,
        # )
        # probs = outputs.predict_probs()[0]
        # boxes = outputs.predict_boxes()[0]
        #
        # # Note: BUTD uses raw RoI predictions,
        # #       we use the predicted boxes instead.
        # # boxes = proposal_boxes[0].tensor
        #
        # # NMS
        # for nms_thresh in np.arange(0.5, 1.0, 0.1):
        #     instances, ids = fast_rcnn_inference_single_image(
        #         boxes, probs, image.shape[1:],
        #         score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
        #     )
        #     if len(ids) == NUM_OBJECTS:
        #         break
        #
        # instances = detector_postprocess(instances, raw_height, raw_width)
        # roi_features = feature_pooled[ids].detach()
        # print(instances)
        #
        # return instances, roi_features


def show_bbox_labels(im):
    pred = instances.to('cpu')
    v = Visualizer(im[:, :, :], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(pred)
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()

    print('instances:\n', instances)
    print()
    print('boxes:\n', instances.pred_boxes)
    print()
    print('Shape of features:\n', features.shape)


def generate_segmentation_file(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    for idx, v in enumerate(imgs_anns.values()):
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        # Because we only have one object category (balloon) to train,
        # 1 is the category of the background
        segmentation = np.ones((height, width), dtype=np.uint8)
        annos = v["regions"]
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = np.array(poly, np.int32)
            category_id = 0
            # category_id = 255  # change to 255 for better visualisation
            cv2.fillPoly(segmentation, [poly], category_id)
            output = os.path.join(img_dir, "segmentation", v["filename"])
            cv2.imwrite(output, segmentation)


# The following is modification of Detectron2 Beginner's Tutorial.
# Cf https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        # Pixel-wise segmentation
        record["sem_seg_file_name"] = os.path.join(img_dir, "segmentation", v["filename"])

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                # "Things" are well-defined countable objects,
                # while "stuff" is amorphous something with a different label than the background.
                "isthing": True,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def visualise_dataset(d: str = "train"):
    dataset_dicts = get_balloon_dicts(os.path.join("balloon", d))
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)

        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def visualise_prediction(predictor, im):
    sem_seg = predictor(im)["sem_seg"]

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    v.save("../output/v_segments.jpg")

    # panoptic segmentation result
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()


def init_cfg(config_file: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    return cfg


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def get_predictor(cfg):
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    # cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    predictor = DefaultPredictor(cfg)
    return predictor


def extract_feature_and_show(predictor, im, sem_seg):
    # box_features, mask_features = doit(predictor, im)

    # get the first channel feature map of instances
    # feature = get_single_feature(sem_seg)
    get_single_feature(sem_seg)

    # visualize the feature map
    # image = visualise_feature_to_img(feature)
    #
    # plt.imshow(image)
    # plt.show()


def get_single_feature(features):
    # extract first channel to output the feature
    features = features[0, :, :, :]
    print(features.shape)  # torch.Size([256, 7, 7])

    # only output the first channel of feature maps
    for i in range(54):
        feature = features[i]
        print(feature.shape)  # torch.Size([7, 7])

    # feature = feature.view(feature.shape[1], feature.shape[2])
    # print(feature.shape)  # torch.Size([112, 112])
    # return feature

    # visualize the feature map
        image = visualise_feature_to_img(feature)
        plt.imshow(image)
        plt.show()


def visualise_feature_to_img(feature):
    # to numpy, numpy不能读取CUDA tensor 需要将它转化为 CPU tensor
    feature = feature.data.cpu().numpy()

    # 将tensor转为numpy，然后归一化到[0, 1]
    # use sigmod to [0,1],对于归一化到[0,1]的部分用了sigmod方法:
    feature = 1.0 / (1 + np.exp(-1 * feature))

    # to [0,255]，最后乘255，使得范围为[0, 255]
    image = np.round(feature * 255)

    # 得到灰度图像并保存
    # cv2.imwrite('../output/feature_visualise.jpg', image)

    return image


if "__main__" == __name__:
    # input
    im = cv2.imread("../input/v.jpg")
    plt.imshow(im[:, :, ::-1])
    plt.show()

    # Setup configuration
    #cfg = init_cfg("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # instance segmentation via mask_rcnn
    cfg = init_cfg("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")  # panoptic segmentation via panoptic FPN

    # Check result
    predictor = get_predictor(cfg)
    outputs = predictor(im)
    print(outputs)
    print(len(outputs["instances"]))  # 55
    print(len(outputs["sem_seg"]))  # 54

    # extract feature
    sem_seg = doit(im)
    print(sem_seg)  # torch.Size([1, 54, 800, 1024])

    # extract feature
    # extract_feature_and_show(predictor, im, sem_seg)

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.SEGMENTATION)
    x = outputs["sem_seg"].argmax(dim=0)
    v = v.draw_sem_seg(x.to("cpu"))

    # v = Visualizer(im[:, :, ::-1],
    #                metadata=balloon_metadata,
    #                scale=0.5,
    #                instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
    #                )
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()
