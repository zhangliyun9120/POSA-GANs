import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, json,  random
from matplotlib import pyplot as plt
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


class FeatureVisualization():
    def __init__(self, img_path, selected_layer):
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.pretrained_model = models.vgg16(pretrained=True).features

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.process_image()
        print(input.shape)  # torch.Size([1, 3, 224, 224])

        x = input
        for index, layer in enumerate(self.pretrained_model):
            x = layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features = self.get_feature()
        print(features.shape)  # torch.Size([1, 128, 112, 112])

        # only output the first channel feature from selected feature map
        feature = features[:, 0, :, :]
        print(feature.shape)  # torch.Size([1, 112, 112])

        feature = feature.view(feature.shape[1], feature.shape[2])
        print(feature.shape)  # torch.Size([112, 112])

        return feature

    def save_feature_to_img(self):
        # to numpy
        feature = self.get_single_feature()
        feature = feature.data.numpy()

        # 将tensor转为numpy，然后归一化到[0, 1]
        # use sigmod to [0,1],对于归一化到[0,1]的部分用了sigmod方法:
        feature = 1.0 / (1 + np.exp(-1 * feature))

        # to [0,255]，最后乘255，使得范围为[0, 255]
        feature = np.round(feature * 255)
        # print(feature[0])

        # 得到灰度图像并保存
        cv2.imwrite('../output/home_vgg16_5th_feature.jpg', feature)


if __name__=='__main__':

    # 输入图片, extract the whole model:
    myClass = FeatureVisualization('../input/home.jpg', 5)
    print(myClass.pretrained_model)

    # extract the selected layer feature and
    # save the processed feature to a gray image from a certain layer of model
    myClass.save_feature_to_img()
