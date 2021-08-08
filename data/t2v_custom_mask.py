#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
# import random
# from collections import defaultdict

import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset

from utils.data import ImageFolder, tensor_to_PIL, PIL_to_tensor, PIL_image_loader

import PIL
from PIL import Image
import json
from torch.utils.data import DataLoader

import torchvision.transforms as T

from utils.data import imagenet_preprocess, Resize

import cv2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# panoptic segmentation:
def get_panoptic_data(dataset_folder):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)

    image_id_to_objects = []

    # img is PIL Image format, so we just use path
    for id, (img, im_path) in enumerate(dataset_folder):
        im = cv2.imread(im_path)
        im = cv2.resize(im, img.size, interpolation=cv2.INTER_AREA)

        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

        # object_data map
        object_data = {
            # 'image_id': 0,
            # 'image_path': None,
            # 'object_class': [],
            'category_id': [],
            'bbox': [],
            'mask': [],
        }

        object_data_out = v.crop_panoptic_seg_predictions(object_data, panoptic_seg.to("cpu"), segments_info)

        image_id_to_objects.append(object_data_out)

    print('dataset has %d images' % (id + 1))

    return image_id_to_objects


class T2VSceneGraphDataset(Dataset):
    def __init__(self, dataset_t, dataset_v, image_id_to_objects, image_size=(256, 256), normalize_images=True):
        super(T2VSceneGraphDataset, self).__init__()

        self.normalize_images = normalize_images

        self.set_image_size(image_size)

        # with open(panoptic_coco_categories, 'r') as f:
        #     categories_list = json.load(f)
        # categegories = {category['id']: category for category in categories_list}  # 134
        # categegories = 134

        self.image_id_to_objects = image_id_to_objects

        # [] is sequential tuple format, {} is un-sequential map format
        self.image_ids = []
        self.image_id_to_filename_t = {}
        self.image_id_to_filename_v = {}
        self.image_id_to_size = {}

        # # load t2V dataset dir to images data
        # dataset_v = ImageFolder(image_dir_v, image_size, transform=None, return_paths=True)
        #
        # # deploy panoptic net in V and crop, then align to T
        self.image_id_to_objects = get_panoptic_data(dataset_v)

        # Thermal
        # dataset_t = ImageFolder(image_dir_t, image_size, transform=None, return_paths=True)
        for id, ((img_t, im_path_t), (_, im_path_v)) in enumerate(zip(dataset_t, dataset_v)):
            image_id = id
            filename_t = im_path_t
            filename_v = im_path_v
            width, height = img_t.size
            self.image_ids.append(image_id)
            self.image_id_to_filename_t[image_id] = filename_t
            self.image_id_to_filename_v[image_id] = filename_v
            self.image_id_to_size[image_id] = (width, height)

        # self.vocab['pred_idx_to_name'] = [
        #     '__in_image__',
        #     'left of',
        #     'right of',
        #     'above',
        #     'below',
        #     'inside',
        #     'surrounding',
        # ]
        # self.vocab['pred_name_to_idx'] = {}
        # for idx, name in enumerate(self.vocab['pred_idx_to_name']):
        #     self.vocab['pred_name_to_idx'][name] = idx

    def set_image_size(self, image_size):
        # print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]

        filename_t = self.image_id_to_filename_t[image_id]
        filename_v = self.image_id_to_filename_v[image_id]
        fname = os.path.basename(filename_t)
        (W, H) = self.image_id_to_size[image_id]
        # image_path = os.path.join(self.image_dir, filename)
        with open(filename_t, 'rb') as f_t:
            with PIL.Image.open(f_t) as image_t:
                # image.show()
                img_t = image_t.resize((W, H), Image.ANTIALIAS)
                img_t = self.transform(img_t.convert('RGB'))
                # img = tensor_to_PIL(img)
                # img.show()
        with open(filename_v, 'rb') as f_v:
            with PIL.Image.open(f_v) as image_v:
                # image.show()
                img_v = image_v.resize((W, H), Image.ANTIALIAS)
                img_v = self.transform(img_v.convert('RGB'))
                # img = tensor_to_PIL(img)
                # img.show()

        boxes, masks = [], []
        object_data = self.image_id_to_objects[image_id]
        objs = torch.LongTensor(object_data['category_id'])  # int64
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

        # shuffle objs
        # O = len(objs)
        # rand_idx = list(range(O))
        # random.shuffle(rand_idx)
        #
        # objs = [objs[i] for i in rand_idx]
        # boxes = [boxes[i] for i in rand_idx]
        # masks = [masks[i] for i in rand_idx]
        #
        # objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        # img = tensor_to_PIL(img)
        # img.show()

        return img_t, img_v, objs, boxes, masks, fname


def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    """
    all_imgs_t, all_imgs_v, all_objs, all_boxes, all_masks, all_obj_to_img = [], [], [], [], [], []

    for i, (img_t, img_v, objs, boxes, masks, fname) in enumerate(batch):
        all_imgs_t.append(img_t[None])
        all_imgs_v.append(img_v[None])
        O = objs.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))

    # cat还可以把list中的tensor拼接起来
    all_imgs_t = torch.cat(all_imgs_t)
    all_imgs_v = torch.cat(all_imgs_v)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_obj_to_img = torch.cat(all_obj_to_img)

    out = (all_imgs_t, all_imgs_v, all_objs, all_boxes, all_masks, all_obj_to_img, fname)

    # print("coco_collate_fn out: ", all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img)

    return out


def get_dataloader(batch_size=1, T2V_DIR='../datasets/thermal2visible_256x256', is_training=True):

    # default mode is training:
    if is_training:
        image_T_dir = os.path.join(T2V_DIR, 'trainT')
        image_V_dir = os.path.join(T2V_DIR, 'trainV')
    else:
        image_T_dir = os.path.join(T2V_DIR, 'testT')
        image_V_dir = os.path.join(T2V_DIR, 'testV')

    # panoptic_coco_categories = './panoptic_coco_categories.json'

    image_size = (256, 256)
    # include_other = True
    # include_relationships = False

    # load t2V dataset dir to images data
    dataset_t = ImageFolder(image_T_dir, image_size, transform=None, return_paths=True)
    dataset_v = ImageFolder(image_V_dir, image_size, transform=None, return_paths=True)

    # deploy panoptic net in V and crop, then align to T
    image_id_to_objects = get_panoptic_data(dataset_v)

    # build datasets
    dset_kwargs = {
        'dataset_t': dataset_t,
        'dataset_v': dataset_v,
        'image_id_to_objects': image_id_to_objects,
        # 'panoptic_coco_categories': panoptic_coco_categories,
        'image_size': image_size,
        'normalize_images': True,
        # 'include_other': include_other,
        # 'include_relationships': include_relationships,
    }

    # dset = T2VSceneGraphDataset(**dset_kwargs)
    # num_imgs = len(dset)
    # print('training dataset has %d images' % num_imgs)
    #
    # # assert train_T_dset.vocab == train_V_dset.vocab
    # #
    # # vocab = json.loads(json.dumps(train_T_dset.vocab))

    # build dataloader
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': False,
        'collate_fn': coco_collate_fn,
    }

    # normal way:
    dataloader = DataLoader(dataset=DatasetFromFolder(**dset_kwargs), **loader_kwargs)

    # # distributed way for multi-gpus:
    # dataloader = DataLoader(dataset=DatasetFromFolder(**dset_kwargs), **loader_kwargs, pin_memory=True,
    #                         sampler=DistributedSampler(dataset=DatasetFromFolder(**dset_kwargs)))

    return dataloader


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_t, dataset_v, image_id_to_objects, image_size=(256, 256), normalize_images=True):
        super(DatasetFromFolder, self).__init__()

        self.normalize_images = normalize_images

        self.set_image_size(image_size)

        self.image_id_to_objects = image_id_to_objects

        # [] is sequential tuple format, {} is un-sequential map format
        self.image_ids = []
        self.image_id_to_filename_t = {}
        self.image_id_to_filename_v = {}
        self.image_id_to_size = {}

        for id, ((img_t, im_path_t), (img_v, im_path_v)) in enumerate(zip(dataset_t, dataset_v)):
            image_id = id
            filename_t = im_path_t
            filename_v = im_path_v
            width, height = img_t.size
            self.image_ids.append(image_id)
            self.image_id_to_filename_t[image_id] = filename_t
            self.image_id_to_filename_v[image_id] = filename_v
            self.image_id_to_size[image_id] = (width, height)

    def set_image_size(self, image_size):
        # print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        filename_t = self.image_id_to_filename_t[image_id]
        filename_v = self.image_id_to_filename_v[image_id]
        fname = os.path.basename(filename_t)
        fname = fname.split('.')[0]
        (W, H) = self.image_id_to_size[image_id]

        with open(filename_t, 'rb') as f_t:
            with PIL.Image.open(f_t) as image_t:
                image_t = image_t.resize((W, H), Image.ANTIALIAS)
                img_t = self.transform(image_t.convert('RGB'))
                # img = tensor_to_PIL(img)
                # img.show()
        with open(filename_v, 'rb') as f_v:
            with PIL.Image.open(f_v) as image_v:
                image_v = image_v.resize((W, H), Image.ANTIALIAS)
                img_v = self.transform(image_v.convert('RGB'))
                # img = tensor_to_PIL(img)
                # img.show()

        boxes, masks = [], []
        object_data = self.image_id_to_objects[image_id]
        objs = torch.LongTensor(object_data['category_id'])  # int64
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

        return img_t, img_v, objs, boxes, masks, fname
        # return img_t, img_v, fname


if __name__ == '__main__':

    # 创建训练以及测试得数据迭代器
    dataloader = get_dataloader(batch_size=1, T2V_DIR='../datasets/thermal2visible_256x256_test', is_training=True)

    # test reading data
    for i, batch in enumerate(dataloader):

        img_t, img_v, objs, boxes, masks, obj_to_img, fname = batch

        print(img_t.shape, img_v.shape, objs.shape, boxes.shape, masks.shape, obj_to_img.shape, fname)

        # img_t, img_v, fname = batch
        #
        # print(img_t.shape, img_v.shape, fname)

