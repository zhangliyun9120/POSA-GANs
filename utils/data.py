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
import torchfile
import torch
import torch.nn as nn
import PIL
import cv2
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data

from torch.autograd import Variable
import os.path
from torch.utils.data import DataLoader
from torchvision import transforms as T

from torch.optim import lr_scheduler
import torch.nn.init as init
import time
import math
import torchvision.utils as vutils

from torchvision import models

from graphviz import Digraph

from imageio import imwrite

from pathlib import Path
from utils.html import HTML

from utils.pytorch_ssim import ssim
from utils.pytorch_lpips import util_of_lpips

import torch.utils.data.distributed as distributed


# normalization image to (0,1):
IMAGENET_MEAN_01 = [0.485, 0.456, 0.406]
IMAGENET_STD_01 = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN_01 = [-m for m in IMAGENET_MEAN_01]
INV_IMAGENET_STD_01 = [1.0 / s for s in IMAGENET_STD_01]

# normalization image to (-1,1):
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

# loader使用torchvision中自带的transforms函数
loader = T.Compose([T.ToTensor()])
unloader = T.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 1, PIL读取图片转化为Tensor: # 输入图片地址 # 返回tensor变量
def PIL_image_loader(image_path):
    image = Image.open(image_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# 2, 将PIL图片转化为Tensor: # 输入PIL格式图片 # 返回tensor变量
def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# 3, Tensor转化为PIL图片: # 输入tensor变量 # 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def get_all_data_loaders(conf):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    # change the size of dataset
    if 'new_size' in conf:
        new_size_a = new_size_b = conf['new_size']
    else:
        new_size_a = conf['new_size_a']
        new_size_b = conf['new_size_b']
    height = conf['crop_image_height']
    width = conf['crop_image_width']

    # further to get train and test dataset
    if 'data_root' in conf:
        train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'trainA'), batch_size, True,
                                              new_size_a, height, width, num_workers, True)
        test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], 'testA'), batch_size, False,
                                             new_size_a, new_size_a, new_size_a, num_workers, True)
        train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'trainB'), batch_size, True,
                                              new_size_b, height, width, num_workers, True)
        test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], 'testB'), batch_size, False,
                                             new_size_b, new_size_b, new_size_b, num_workers, True)

    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
    transform_list = [T.ToTensor(),
                      T.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [T.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [T.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [T.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = T.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


def get_config(config):
    with open(config, 'r') as stream:
        # YAML 5.1版本后弃用了yaml.load(file)这个用法，因为觉得很不安全，5.1版本之后就修改了需要指定Loader，
        # 通过默认加载器（FullLoader）禁止执行任意函数，该load函数也变得更加安全
        return yaml.load(stream, Loader=yaml.FullLoader)


def default_loader(path, image_size):
    img = Image.open(path)
    img1 = img.resize(image_size, Image.ANTIALIAS)
    img1 = img1.convert('RGB')
    return img1


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, image_size, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.image_size = image_size
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path, self.image_size)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


def imagenet_preprocess_01():
    return T.Normalize(mean=IMAGENET_MEAN_01, std=IMAGENET_STD_01)


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess_01(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD_01),
        T.Normalize(mean=INV_IMAGENET_MEAN_01, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_deprocess_batch_01(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess_01(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def imagenet_deprocess_batch(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255).byte()
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de


def img_save(name, img, i, save_dir):
    # save normalized image format to normal image format:
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img_path = os.path.join(save_dir, 'iter{}_{}.png'.format(i + 1, name))
    print("Generated image: {}".format(img_path))
    imwrite(img_path, img)


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


def show_single_feature(features, text):
    for i in range(features.size(0)):
        feature = features[i, :, :, :]
        feature = feature[0, :, :]  # only extract o dimension
        # visualize the feature map
        image = visualise_feature_to_img(feature)
        if i == 4:
            plt.imshow(image)
            plt.show()
            path = "../images/feature_" + text + "_" + str(i) + "_" + ".jpg"
            cv2.imwrite(path, image)
            break


def image_objects_show(imgs):
    for i in range(imgs.size(0)):
        img = imgs[i, :, :, :].unsqueeze(0)
        # show cropped image clip
        img = imagenet_deprocess_batch(img)
        img = img[0].numpy().transpose(1, 2, 0)
        img = Image.fromarray(img)
        plt.imshow(img)
        plt.show()
        # img.save("../images/image_" + text + "_" + str(i) + "_" + ".jpg")


def image_show(img, i, text):
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()
    # img.save("../images/image_" + text + "_" + str(i) + "_" + ".jpg")


def image_with_bbox_mask_show(img, box, mask, i):
    box = box * img.size(2)  # recover the box value from [0,1] to [0, 255]
    x0, y0, x1, y1 = box  # extract coordinate
    mask = mask.cpu().clone()  # convert cuda to cpu

    # convert img to numpy and then to CV data
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2

    # draw mask on image
    if i == 0:
        color_mask = np.array([255, 129, 0], dtype=np.uint8)  # orange
    elif i == 3:
        color_mask = np.array([84, 130, 53], dtype=np.uint8)  # pale green
    elif i == 4:
        color_mask = np.array([192, 0, 0], dtype=np.uint8)  # crimson
    else:
        color_mask = np.array([0, 255, 0], dtype=np.uint8)  # green
    mask = mask.numpy().astype(np.bool)
    img[mask] = img[mask] * 0.5 + color_mask * 0.5

    # draw bbox on image
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)  # red color

    # convert whole image to Image and show it
    if i == 4:
        img = Image.fromarray(img)
        plt.imshow(img)
        plt.show()
        img.save("../images/image_bbox_mask_" + str(i) + "_" + ".jpg")


def binary_image_with_bbox_mask_show(img, box, mask, i):
    box = box * img.size(2)  # recover the box value from [0,1] to [0, 255]
    x0, y0, x1, y1 = box  # extract coordinate
    mask = mask.cpu().clone()  # convert cuda to cpu

    # convert img to numpy and then to CV data
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2

    # # draw mask on image
    # if i == 0:
    #     color_mask = np.array([127, 81, 127], dtype=np.uint8)  # orange  road
    # elif i == 1:
    #     color_mask = np.array([113, 135, 63], dtype=np.uint8)  # pale green  building
    # elif i == 2:
    #     color_mask = np.array([113, 137, 63], dtype=np.uint8)  # pale green  tree
    # elif i == 3:
    #     color_mask = np.array([84, 130, 53], dtype=np.uint8)  # pale green sky
    # elif i == 4:
    #     color_mask = np.array([68, 125, 230], dtype=np.uint8)  # crimson car
    # else:
    #     color_mask = np.array([0, 255, 0], dtype=np.uint8)  # green
    # mask = mask.numpy().astype(np.bool)
    # img[mask] = img[mask] * 0.5 + color_mask * 0.5

    # creat a binary image same to raw img
    height = img.shape[0]
    weight = img.shape[1]
    for row in range(height):  # 遍历高
        for col in range(weight):  # 遍历宽
            if not mask[row, col]:  # False
                r = img[row, col, 0] = 0
                g = img[row, col, 1] = 0
                b = img[row, col, 2] = 0

    # draw bbox on image, -1表示的是填充矩形的意思
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (127, 81, 127), -1)  # red color

    # convert whole image to Image and show it
    # if i == 0:
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()
    img.save("../images/binary_image_bbox_mask_" + str(i) + "_" + ".jpg")


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            H, W = size
            self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


def unpack_var(v):
    if isinstance(v, torch.autograd.Variable):
        return v.data
    return v


def split_graph_batch(triples, obj_data, obj_to_img, triple_to_img):
    triples = unpack_var(triples)
    obj_data = [unpack_var(o) for o in obj_data]
    obj_to_img = unpack_var(obj_to_img)
    triple_to_img = unpack_var(triple_to_img)

    triples_out = []
    obj_data_out = [[] for _ in obj_data]
    obj_offset = 0
    N = obj_to_img.max() + 1
    for i in range(N):
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)

        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        triples_out.append(cur_triples)

        for j, o_data in enumerate(obj_data):
            cur_o_data = None
            if o_data is not None:
                cur_o_data = o_data[o_idxs]
            obj_data_out[j].append(cur_o_data)

        obj_offset += o_idxs.size(0)

    return triples_out, obj_data_out


def write_one_row_html(html_file, iterations, img_filename, all_size, label):
    # html_file.write("<h3>Iteration: %d  (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("<h3>Iteration: %d  (%s)</h3>" % (iterations, label))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_directory, all_size=512):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>Comparison</h3>")
    write_one_row_html(html_file, iterations, '%s/img_t_%08d.png' % (image_directory, iterations), all_size, "Real_Thermal")
    write_one_row_html(html_file, iterations, '%s/img_t2v_%08d.png' % (image_directory, iterations), all_size, "Fake_Visible")
    write_one_row_html(html_file, iterations, '%s/img_v_%08d.png' % (image_directory, iterations), all_size, "Real_Visible")
    html_file.write("</body></html>")
    html_file.close()


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


# TV Loss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# VGG Features matching
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def compute_vgg_loss(vgg, img, target):
    img_vgg = vgg_preprocess(img)
    target_vgg = vgg_preprocess(target)
    img_fea = vgg(img_vgg)
    target_fea = vgg(target_vgg)
    # 使用正则化的方式
    instancenorm = nn.InstanceNorm2d(512, affine=False)
    return torch.mean((instancenorm(img_fea) - instancenorm(target_fea)) ** 2)


def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.to_relu_5_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        for x in range(23, 30):
            self.to_relu_5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        h = self.to_relu_5_3(h)
        h_relu_5_3 = h
        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3, h_relu_5_3)
        return out


def get_scheduler(optimizer, config, iterations=-1):
    if config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma, last_epoch=iterations)
    else:
        scheduler = None  # constant scheduler
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


# 画pytorch模型图，以及参数计算
def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


def image_save(name, img, save_dir):
    # save normalized image format to normal image format:
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    img_path = os.path.join(save_dir, '{}.jpg'.format(name))
    print("Generated image: {}".format(img_path))
    imwrite(img_path, img)


def html_save(n, save_dir, config, fname_list):
    # save images to website
    img_path = os.path.join('/home/liyun/Desktop/Challenge1_experiment/OL-GANs/experiment4/results', 'images')
    html = HTML(save_dir, 'experiment')
    for i in range(n, 0, -config.image_step):
        html.add_header('[%d]' % i)
        ims = []
        ims.append(os.path.join(img_path, 'img_%08d_t_%s.jpg' % (i, fname_list[i-1])))
        ims.append(os.path.join(img_path, 'img_%08d_t2v_%s.jpg' % (i, fname_list[i-1])))
        ims.append(os.path.join(img_path, 'img_%08d_v_%s.jpg' % (i, fname_list[i-1])))
        texts = ['Thermal', 'Fake Visible', 'Visible']
        html.add_images(ims, texts)
    html.save()


def image_save_test(name, img, save_dir):
    # save normalized image format to normal image format:
    img = imagenet_deprocess_batch_01(img)
    img = img[0].numpy().transpose(1, 2, 0)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    img_path = os.path.join(save_dir, '{}.png'.format(name))
    print("Generated image: {}".format(img_path))
    imwrite(img_path, img)


def html_save_test(n, save_dir, fname):
    # save images to website
    img_path = os.path.join('/home/liyun/Desktop/Challenge1_experiment/SF-GANs/experiment7/tests', 'images')
    html = HTML(save_dir, 'experiment')
    for n in range(n, 0, -1):
        html.add_header('[%d] %s' % (n, fname))
        ims = []
        ims.append(os.path.join(img_path, 'img%s_t.png' % fname))
        ims.append(os.path.join(img_path, 'img%s_t2v.png' % fname))
        ims.append(os.path.join(img_path, 'img%s_v.png' % fname))
        texts = ['Thermal', 'Fake Visible', 'Visible']
        html.add_images(ims, texts)
    html.save()


def validation_methods(sr, hr):
    batch_mse = ((sr - hr) ** 2).data.mean()
    batch_ssim = ssim(sr, hr).item()
    psnr = 10 * math.log10(1 / (batch_mse / sr.size(0)))

    return batch_mse, batch_ssim, psnr


def calculation_lpips(img1_path, img2_path):
    ## Initializing the model
    loss_fn = util_of_lpips(net='alex', use_gpu=True)
    # Compute distance
    lpips = loss_fn.calc_lpips(img1_path, img2_path)

    return lpips


def convert_normalize(tensor_image):
    # [-1, 1] to [0, 1]
    # step 1: convert it to [0 ,2]
    tensor_image = tensor_image + 1
    # step 2: convert it to [0 ,1]
    tensor_image = tensor_image - tensor_image.min()
    tensor_image_0_1 = tensor_image / (tensor_image.max() - tensor_image.min())

    return tensor_image_0_1


def convert_transform(tensor_image):
    # [-1, 1] to [0, 1]
    img = imagenet_deprocess_batch(tensor_image)
    img = img[0].numpy().transpose(1, 2, 0)
    img = Image.fromarray(img)
    # plt.imshow(img)
    # plt.show()
    image_size = (256, 256)
    transform = [Resize(image_size), T.ToTensor()]
    transform = T.Compose(transform)
    img = transform(img.convert('RGB'))

    return img


def image_show_normal(img):
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()


def image_show_test(img):
    img = imagenet_deprocess_batch_01(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()
