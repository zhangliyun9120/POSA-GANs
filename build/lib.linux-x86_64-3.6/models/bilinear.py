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

import torch
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms as T


"""
Functions for performing differentiable bilinear cropping of images, for use in
the object discriminator
"""


def crop_bbox_batch(feats, bbox, bbox_to_feats, HH, WW=None, backend='cudnn'):
    """
    Inputs:
    - feats: FloatTensor of shape (N, C, H, W)
    - bbox: FloatTensor of shape (B, 4) giving bounding box coordinates
    - bbox_to_feats: LongTensor of shape (B,) mapping boxes to feature maps;
      each element is in the range [0, N) and bbox_to_feats[b] = i means that
      bbox[b] will be cropped from feats[i].
    - HH, WW: Size of the output crops

    Returns:
    - crops: FloatTensor of shape (B, C, HH, WW) where crops[i] uses bbox[i] to
      crop from feats[bbox_to_feats[i]].
    """
    if backend == 'cudnn':
        return crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW)
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None: WW = HH
    dtype, device = feats.dtype, feats.device
    crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]
        cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
        crops[idx] = cur_crops
    return crops


def _invperm(p):
    N = p.size(0)
    eye = torch.arange(0, N).type_as(p)
    pp = (eye[:, None] == p).nonzero()[:, 1]
    return pp


def crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW=None):
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None: WW = HH
    dtype = feats.data.type()

    feats_flat, bbox_flat, all_idx = [], [], []
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero(as_tuple=False)
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]

        feats_flat.append(cur_feats)
        bbox_flat.append(cur_bbox)
        all_idx.append(idx)

    feats_flat = torch.cat(feats_flat, dim=0)
    bbox_flat = torch.cat(bbox_flat, dim=0)
    crops = crop_bbox(feats_flat, bbox_flat, HH, WW, backend='cudnn')

    # If the crops were sequential (all_idx is identity permutation) then we can
    # simply return them; otherwise we need to permute crops by the inverse
    # permutation from all_idx.
    all_idx = torch.cat(all_idx, dim=0)
    eye = torch.arange(0, B).type_as(all_idx)
    if (all_idx == eye).all():
        return crops
    return crops[_invperm(all_idx)]


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if WW is None: WW = HH
    if backend == 'cudnn':
        # Change box from [0, 1] to [-1, 1] coordinate system
        bbox = 2 * bbox - 1
    x0, y0 = bbox[:, 0], bbox[:, 1]
    x1, y1 = bbox[:, 2], bbox[:, 3]
    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    if backend == 'jj':
        return bilinear_sample(feats, X, Y)
    elif backend == 'cudnn':
        grid = torch.stack([X, Y], dim=3)
        return F.grid_sample(feats, grid)


def uncrop_bbox(feats, bbox, H, W=None, fill_value=0):
    """
    Inverse operation to crop_bbox; construct output images where the feature maps
    from feats have been reshaped and placed into the positions specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, HH, WW)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - H, W: Size of output.
    - fill_value: Portions of the output image that are outside the bounding box
      will be filled with this value.

    Returns:
    - out: Tensor of shape (N, C, H, W) where the portion of out[i] given by
      bbox[i] contains feats[i], reshaped using bilinear sampling.
    """
    N, C = feats.size(0), feats.size(1)
    assert bbox.size(0) == N
    assert bbox.size(1) == 4
    if W is None: H = W

    x0, y0 = bbox[:, 0], bbox[:, 1]
    x1, y1 = bbox[:, 2], bbox[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    x1 = x1.contiguous().view(N, 1).expand(N, H)
    ww = ww.view(N, 1).expand(N, H)

    y0 = y0.contiguous().view(N, 1).expand(N, W)
    y1 = y1.contiguous().view(N, 1).expand(N, W)
    hh = hh.view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).to(feats)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).to(feats)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    # For ByteTensors, (x + y).clamp(max=1) gives logical_or
    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)
    out_mask = (X_out_mask + Y_out_mask).clamp(max=1)
    out_mask = out_mask.view(N, 1, H, W).expand(N, C, H, W)

    X = X.view(N, 1, W).expand(N, H, W)
    Y = Y.view(N, H, 1).expand(N, H, W)

    out = bilinear_sample(feats, X, Y)
    out[out_mask] = fill_value
    return out


def bilinear_sample(feats, X, Y):
    """
    Perform bilinear sampling on the features in feats using the sampling grid
    given by X and Y.

    Inputs:
    - feats: Tensor holding input feature map, of shape (N, C, H, W)
    - X, Y: Tensors holding x and y coordinates of the sampling
      grids; both have shape shape (N, HH, WW) and have elements in the range [0, 1].
    Returns:
    - out: Tensor of shape (B, C, HH, WW) where out[i] is computed
      by sampling from feats[idx[i]] using the sampling grid (X[i], Y[i]).
    """
    N, C, H, W = feats.size()
    assert X.size() == Y.size()
    assert X.size(0) == N
    _, HH, WW = X.size()

    X = X.mul(W)
    Y = Y.mul(H)

    # Get the x and y coordinates for the four samples
    x0 = X.floor().clamp(min=0, max=W - 1)
    x1 = (x0 + 1).clamp(min=0, max=W - 1)
    y0 = Y.floor().clamp(min=0, max=H - 1)
    y1 = (y0 + 1).clamp(min=0, max=H - 1)

    # In numpy we could do something like feats[i, :, y0, x0] to pull out
    # the elements of feats at coordinates y0 and x0, but PyTorch doesn't
    # yet support this style of indexing. Instead we have to use the gather
    # method, which only allows us to index along one dimension at a time;
    # therefore we will collapse the features (BB, C, H, W) into (BB, C, H * W)
    # and index along the last dimension. Below we generate linear indices into
    # the collapsed last dimension for each of the four combinations we need.
    y0x0_idx = (W * y0 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x0_idx = (W * y1 + x0).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y0x1_idx = (W * y0 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)
    y1x1_idx = (W * y1 + x1).view(N, 1, HH * WW).expand(N, C, HH * WW)

    # Actually use gather to pull out the values from feats corresponding
    # to our four samples, then reshape them to (BB, C, HH, WW)
    feats_flat = feats.view(N, C, H * W)
    v1 = feats_flat.gather(2, y0x0_idx.long()).view(N, C, HH, WW)
    v2 = feats_flat.gather(2, y1x0_idx.long()).view(N, C, HH, WW)
    v3 = feats_flat.gather(2, y0x1_idx.long()).view(N, C, HH, WW)
    v4 = feats_flat.gather(2, y1x1_idx.long()).view(N, C, HH, WW)

    # Compute the weights for the four samples
    w1 = ((x1 - X) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w2 = ((x1 - X) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w3 = ((X - x0) * (y1 - Y)).view(N, 1, HH, WW).expand(N, C, HH, WW)
    w4 = ((X - x0) * (Y - y0)).view(N, 1, HH, WW).expand(N, C, HH, WW)

    # Multiply the samples by the weights to give our interpolated results.
    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return out


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def masking_feature(boxes, fm_objs, masks):
    """
    Inputs:
        - boxes: Tensor of shape (b, num_o, 4) giving bounding boxes in the format
            [x0, y0, x1, y1] in the [0, 1] coordinate space
        - masks: Tensor of shape (b, num_o, M, M) giving binary masks for each object
        - H, W: Size of the output image.
    Returns:
        - out: Tensor of shape (N, num_o, H, W)
    """
    cls, _ = boxes.size()
    cls, C, M, M = fm_objs.size()
    cls, H, W = masks.size()

    grid = _boxes_to_grid(boxes.view(cls, -1), H, W).float().cuda(device=fm_objs.device)

    img_in = fm_objs.float().view(cls, C, M, M)
    bboxed_feature = F.grid_sample(img_in, grid, mode='bilinear')
    cls, C, S, S = bboxed_feature.size()

    masks = masks.unsqueeze(1)
    print("masks.shape: ", masks.shape)   # torch.Size([cls, 1, 256, 256])
    mask_copy = torch.cat([masks] * C, dim=1)

    # points of feature map will be lost setup to 0, since they are out of mask
    masked_feature = bboxed_feature.masked_fill(mask_copy == 0, value=torch.tensor(0))

    return masked_feature


def get_single_feature(features):
    num_o, b, H, W = features.size()
    features = features.view(b * num_o, H, W)
    # extract first channel to output the feature
    features = features[0, :, :, :]
    print(features.shape)  # torch.Size([256, 7, 7])

    # only output the first channel of feature maps
    for i in range(num_o):
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


# normalization image to (-1,1):
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


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


def image_save(name, img, i, save_dir):
    # save normalized image format to normal image format:
    img = imagenet_deprocess_batch(img)
    img = img[0].numpy().transpose(1, 2, 0)
    img_path = os.path.join(save_dir, 'iter{}_{}.png'.format(i + 1, name))
    print("Generated image: {}".format(img_path))
    imwrite(img_path, img)


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid


def compute_transformation_matrix(bbox):
    x, y = bbox[:, 0], bbox[:, 1]
    w, h = bbox[:, 2], bbox[:, 3]

    scale_x = w
    scale_y = h

    t_x = 2 * ((x + 0.5 * w) - 0.5)
    t_y = 2 * ((y + 0.5 * h) - 0.5)

    zeros = torch.cuda.FloatTensor(bbox.shape[0],1).fill_(0)

    transformation_matrix = torch.cat([scale_x.unsqueeze(-1), zeros, t_x.unsqueeze(-1),
                                       zeros, scale_y.unsqueeze(-1), t_y.unsqueeze(-1)], 1).view(-1, 2, 3)

    return transformation_matrix


if __name__ == '__main__':
    import numpy as np
    from scipy.misc import imread, imsave, imresize

    cat = imresize(imread('cat.jpg'), (256, 256))
    dog = imresize(imread('dog.jpg'), (256, 256))
    feats = torch.stack([
        torch.from_numpy(cat.transpose(2, 0, 1).astype(np.float32)),
        torch.from_numpy(dog.transpose(2, 0, 1).astype(np.float32))],
        dim=0)

    boxes = torch.FloatTensor([
        [0, 0, 1, 1],
        [0.25, 0.25, 0.75, 0.75],
        [0, 0, 0.5, 0.5],
    ])

    box_to_feats = torch.LongTensor([1, 0, 1]).cuda()

    feats, boxes = feats.cuda(), boxes.cuda()
    crops = crop_bbox_batch_cudnn(feats, boxes, box_to_feats, 128)
    for i in range(crops.size(0)):
        crop_np = crops.data[i].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        imsave('out%d.png' % i, crop_np)
