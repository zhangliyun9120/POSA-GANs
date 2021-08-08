
import json
import os
import random
import torch

import cv2
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor, launch
from detectron2.config import get_cfg

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image


class feature_extractor:
    '''
    Feature Extractor for detectron2
    '''
    def __init__(self, path=None, output_folder='./output', model=None, pred_thresh=0.5):
        self.pred_thresh = pred_thresh
        self.output_folder = output_folder
        assert path is not None, 'Path should not be none'
        self.path = path
        if model == None:
            self.model = self._build_detection_model()
        else:
            assert model == detectron2.engine.defaults.DefaultPredictor, "model should be 'detectron2.engine.defaults.DefaultPredictor'"#
            self.model = model
            self.model.eval()

    def _build_detection_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (pnumonia)
        # Just run these lines if you have the trained model im memory
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.pred_thresh   # set the testing threshold for this model
        # build model and return
        return DefaultPredictor(cfg)

    def _process_feature_extraction(self, img):  # step 3
        '''
        #predictor.model.roi_heads.box_predictor.test_topk_per_image = 1000
        #predictor.model.roi_heads.box_predictor.test_nms_thresh = 0.99
        #predictor.model.roi_heads.box_predictor.test_score_thresh = 0.0
        #pred_boxes = [x.pred_boxes for x in instances]#can use prediction boxes
        '''
        torch.cuda.empty_cache()
        predictor = self.model
        with torch.no_grad():

            features = predictor.model.backbone(img.tensor)  # Get feature of backbone FPN

            proposals, _ = predictor.model.proposal_generator(img, features, None)  # Get proposal RPN

            results, _ = predictor.model.roi_heads(img, features, proposals, None)  # Get instances of roi_heads

            # instances = predictor.model.roi_heads._forward_box(features, proposals)

        # get proposed boxes + rois + features + predictions
        proposal_boxes = [x.proposal_boxes for x in proposals]
        proposal_rois = predictor.model.roi_heads.box_pooler(
            [features[f] for f in predictor.model.roi_heads.in_features], proposal_boxes)
        box_features = predictor.model.roi_heads.box_head(proposal_rois)
        predictions = predictor.model.roi_heads.box_predictor(box_features)

        # WE CAN USE THE PREDICTION CLS TO FIND TOP SCOREING PROPOSAL BOXES!
        # pred_instances, losses = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_df = pd.DataFrame(predictions[0].softmax(-1).tolist())
        pred_classes = pred_df.iloc[:,:-1].apply(np.argmax, axis=1)  # get predicted classes
        keep = pred_df[pred_df.iloc[:, :-1].apply(
            lambda x: (x > self.pred_thresh)).values].index.tolist()  # list of instances we should keep
        # start subsetting
        box_features = box_features[keep]
        proposal_boxes = proposals[0].proposal_boxes[keep]
        pred_classes = pred_classes[keep]
        probs = pred_df.iloc[keep, :].apply(lambda x: x[np.argmax(x)], axis=1).tolist()

        # ['bbox', 'num_boxes', 'objects', 'image_width', 'image_height', 'cls_prob', 'image_id', 'features']
        # img.image_sizes[0]#h, w
        result = {
            'bbox': proposal_boxes.tensor.to('cpu').numpy(),
            'num_boxes': len(proposal_boxes),
            'objects': pred_classes.to_numpy,
            # 'image_height': img.image_sizes[0][0],
            # 'image_width': img.image_sizes[0][1],
            'cls_prob': np.asarray(probs),  # needs to turn into vector!!!!!!!!!!
            'features': box_features.to('cpu').detach().numpy()
        }
        return result

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        feature["image_id"] = file_base_name
        feature['image_height'] = info['height']
        feature['image_width'] = info['width']
        file_base_name = file_base_name + ".npy"
        np.save(os.path.join(self.output_folder, file_base_name), feature)

    def extract_features(self):  # step 1
        torch.cuda.empty_cache()
        image_dir = self.path
        # print(image_dir)
        if type(image_dir) == pd.core.frame.DataFrame:  # or pandas.core.frame.DataFrame. Iterate over a dataframe
            samples = []
            for idx, row in image_dir.iterrows():  # get better name
                file = row['path']
                try:
                    features, infos = self.get_detectron2_features([file])
                    self._save_feature(file, features, infos[0])
                    samples.append(row)
                except BaseException:  # if no features were found!
                    print('No features were found!')
                    pass
            df = pd.DataFrame(samples)
            # save final csv containing image base names, reports and report locations
            df.to_csv(os.path.join(self.output_folder, 'img_infos.csv'))
        elif os.path.isfile(image_dir):  # if its a single file
            features, infos = self.get_detectron2_features([image_dir])
            self._save_feature(image_dir, features[0], infos[0])
            return features, infos
        else:  # if its a directory
            files = glob.glob(os.path.join(image_dir, "*"))
            for idx, file in enumerate(files):
                try:
                    features, infos = self.get_detectron2_features([file])
                    self._save_feature(file, features, infos[0])
                except BaseException:
                    print('BaseException')
                    pass

    def get_detectron2_features(self, image_paths):  # step 2
        # we have to PREPROCESS the tensor before partially executing it!
        predictor = self.model
        images = []
        image_info = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            img = predictor.transform_gen.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            images.append({"image": img, "height": height, "width": width})  # add new item in list tail
            image_info.append({"image_id": os.path.basename(image_path), "height": height, "width": width})
        imageList = predictor.model.preprocess_image(images)
        # returns features and infos
        return self._process_feature_extraction(imageList), image_info


if "__main__" == __name__:
    # path is image input dir, output_folder is feature output dir
    fe = feature_extractor(path='../input', output_folder='../output')
    fe.extract_features()
