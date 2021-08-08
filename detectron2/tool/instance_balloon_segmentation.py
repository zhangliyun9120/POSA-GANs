'''
Train a balloon instance segmentation detector.
'''

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, torch
import pandas as pd
import glob
from matplotlib import pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image


# NUM_OBJECTS = 36

'''
Register the balloon dataset to detectron2,

following the detectron2 custom dataset tutorial. Here, the dataset is in its custom format, 
therefore we write a function to parse it and prepare it into detectron2's standard format. User should write such 
a function when using a dataset in custom format. See the tutorial for more details.
'''

# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")


class feature_extractor:
    '''
    Feature Extractor for detectron2
    '''
    def __init__(self, path=None, output_folder=None, model=None, pred_thresh=0.5):
        self.pred_thresh = pred_thresh
        self.output_folder = output_folder
        assert path is not None, 'Path should not be none'
        self.path = path
        if model == None:
            self.model = self._build_detection_model()
        else:
            assert model == detectron2.engine.defaults.DefaultPredictor, "model should be 'detectron2.engine.defaults.DefaultPredictor'"
            self.model = model
            self.model.eval()

    def _build_detection_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.SOLVER.IMS_PER_BATCH = 1
        # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (pnumonia)
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
        probs = pred_df.iloc[keep, :].apply(lambda x: x[np.argmax(x)], axis=1).values.tolist()

        # ['bbox', 'num_boxes', 'objects', 'image_width', 'image_height', 'cls_prob', 'image_id', 'features']
        # img.image_sizes[0] # h, w
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
        print(image_dir)
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
            # self._save_feature(image_dir, features, infos)
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
        # we have to preprocess the tensor before partially executing it!
        predictor = self.model
        images = []
        image_info = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            img = predictor.aug.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            images.append({"image": img, "height": height, "width": width})  # add new item in list tail
            image_info.append({"image_id": os.path.basename(image_path), "height": height, "width": width})
        imageList = predictor.model.preprocess_image(images)
        # returns features and infos
        return self._process_feature_extraction(imageList), image_info


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
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def visualise_dataset(d: str):
    dataset_dicts = get_balloon_dicts(os.path.join("balloon", d))
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()


def visualise_prediction(predictor, d: str = "val"):
    dataset_dicts = get_balloon_dicts(os.path.join("balloon", d))
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)

        # extract feature
        extract_feature_and_show(predictor, im)

        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.show()


def init_cfg(config_file: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ("balloon_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 200
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    return cfg


def train(cfg):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def get_predictor(cfg, model_name: str):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor


def doit(predictor, raw_image):
    with torch.no_grad():
        # 1, Get W and H of raw image input
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))

        # 2, Preprocessing raw image:
        # 将图像缩放到固定大小
        image = predictor.aug.get_transform(raw_image).apply_image(raw_image)  # transform the W and H of raw image
        print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # ndarray to tensor and (h,w,c) to (c,h,w)
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)  # get a ImageList of Detectron2

        # 3, Get feature of backbone FPN:
        # 将 transform 后的图像输入到 backbone 模块提取特征图
        # 对于普通的 FasterRCNN 只需要将 feature_map 输入到 rpn 网络生成 proposals 即可。
        # 但是由于加入 FPN，需要将多个 feature_map 逐个输入到 rpn 网络。
        features = predictor.model.backbone(images.tensor)
        print("Backbone Features:", features)  # a dict for backbone feature (p2, p3, p4, p5, p6) layers

        # 4, Get proposal feature of proposal_generator RPN:
        # 然后经过 rpn 模块生成 proposals 和 proposal_losses
        proposals, _ = predictor.model.proposal_generator(images, features)  # list proposed boxes for image regions
        print("Generate proposals with RPN:", proposals)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        # proposal_boxes = [x.proposal_boxes for x in proposals]
        # features = [features[f] for f in predictor.model.roi_heads.in_features]
        # box_features = predictor.model.roi_heads._shared_roi_transform(
        #     features, proposal_boxes
        # )

        # 5, Get instances of roi_heads:
        instances, _ = predictor.model.roi_heads(images, features, proposals)  # list instances [16, 1, 28, 28] object

        # 6, Get box feature of box_pooler after ROIAlign
        in_features = [features[f] for f in predictor.model.roi_heads.in_features]  # Tensor box [16, 256, 7, 7]
        box_features = predictor.model.roi_heads.box_pooler(in_features, [x.pred_boxes for x in instances])

        # 7, Get mask feature of mask_pooler after ROIAlign
        in_features = [features[f] for f in predictor.model.roi_heads.in_features]  # Tensor mask [16, 256, 14, 14]
        mask_features = predictor.model.roi_heads.mask_pooler(in_features, [x.pred_boxes for x in instances])

        return box_features, mask_features

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


def extract_feature_and_show(predictor, im):
    box_features, mask_features = doit(predictor, im)

    # get the first channel feature map of instances
    feature = get_single_feature(box_features)

    # visualize the feature map
    image = visualise_feature_to_img(feature)

    plt.imshow(image)
    plt.show()


def get_single_feature(features):
    # extract first channel to output the feature
    feature = features[0]  # features: torch.Size([cls, 256, 7, 7])
    print(feature.shape)  # torch.Size([256, 7, 7])

    # only output the first channel of feature maps
    feature = feature[0, :, :]
    print(feature.shape)  # torch.Size([7, 7])

    # feature = feature.view(feature.shape[1], feature.shape[2])
    # print(feature.shape)  # torch.Size([112, 112])
    return feature


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
    # 1, create instance segmentation dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    # 2, visualize the annotated dataset is correctly initialised
    # visualise_dataset("train")

    # 3, train the instance segmentation model based on created dataset
    # Setup configuration
    # instance: Mask R-CNN
    cfg = init_cfg("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # Vanilla single-GPU training
    # train(cfg)

    # Multi-GPU training
    '''
    launch(
        train,
        1,  # Number of GPUs per machine
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:1234",
        args=(cfg,),
    )
    '''

    # 4, Inference & evaluation using the trained model on the balloon validation dataset.
    predictor = get_predictor(cfg, "model_final.pth")
    visualise_prediction(predictor, "val")


