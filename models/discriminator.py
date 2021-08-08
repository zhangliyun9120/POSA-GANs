import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import json
import numpy as np
from models.layers import build_cnn
from models.layers import GlobalAvgPool
from models.bilinear import crop_bbox_batch

import torch.backends.cudnn as cudnn

from models.roi_layers import ROIAlign, ROIPool

cudnn.benchmark = True
device = torch.device('cuda:0')


def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Embedding)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _downsample(x):
    return F.avg_pool2d(x, kernel_size=2)


class OptimizedBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.learnable_sc = (dim_in != dim_out) or downsample
        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        h = x
        if self.downsample:
            h = _downsample(x)
        return self.sc(h)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True))

        self.learnable_sc = (dim_in != dim_out) or downsample

        if self.learnable_sc:
            self.sc = nn.Conv2d(dim_in, dim_out, kernel_size=1, padding=0, bias=True)

    def residual(self, x):
        h = x
        h = self.resi(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResBlock_Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock_Downsample, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()
        self.downsample = downsample

    def forward(self, in_feat):
        x = in_feat
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)

    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.c_sc(x)


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, class_num):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(class_num, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class Obj_ResnetRoI(nn.Module):
    def __init__(self, class_num=133, input_dim=3, ch=32):
        super(Obj_ResnetRoI, self).__init__()
        self.class_num = class_num

        self.block1 = BasicBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock_Downsample(ch, ch * 2, downsample=True)
        self.block3 = ResBlock_Downsample(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock_Downsample(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock_Downsample(ch * 8, ch * 8, downsample=False)
        self.classifier_img = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
        self.activation = nn.ReLU()

        # object classification:
        self.classifier_cls = nn.utils.spectral_norm(nn.Linear(ch * 8, class_num))

    def forward(self, crops):
        x = crops
        # 32x32 -> 16x16
        x = self.block1(x)
        # 16x16 -> 8x8
        x = self.block2(x)
        # 8x8 -> 4x4
        x = self.block3(x)
        # 4x4 -> 2x2
        x = self.block4(x)
        x = self.block5(x)

        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))  # (1, 256)
        out_im = self.classifier_img(x)  # (256, 1)

        out_cls = self.classifier_cls(x)  # (256, cls)

        return out_im, out_cls


class ResnetRoI(nn.Module):
    def __init__(self, class_num=134, input_dim=3, ch=64):
        super(ResnetRoI, self).__init__()
        self.class_num = class_num

        self.block1 = BasicBlock(input_dim, ch, downsample=True)
        self.block2 = ResBlock_Downsample(ch, ch * 2, downsample=True)
        self.block3 = ResBlock_Downsample(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock_Downsample(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock_Downsample(ch * 8, ch * 8, downsample=True)
        self.block6 = ResBlock_Downsample(ch * 8, ch * 16, downsample=True)
        self.block7 = ResBlock_Downsample(ch * 16, ch * 16, downsample=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 16.0, int(0))

        self.block_obj4 = ResBlock_Downsample(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock_Downsample(ch * 8, ch * 8, downsample=False)
        self.block_obj6 = ResBlock_Downsample(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(self.class_num, ch * 16))

        # object classification:
        self.classifier_cls = nn.utils.spectral_norm(nn.Linear(ch * 16, class_num))

    def forward(self, images, bbox, label):
        bbox = bbox.unsqueeze(0)  # (cls, 4) -> (1, cls, 4)
        label = label.unsqueeze(0).unsqueeze(-1)  # (cls,) -> (1, cls, 1)
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0), 1, 1).expand(-1, bbox.size(1), -1).float()
        # bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        # bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)
        # idx = (label != 0).nonzero(as_tuple=False).view(-1)
        # bbox = bbox[idx]
        # label = label[idx]
        cls = bbox.size(0)

        # (3,256x256) -> (64,128x128)
        x = self.block1(images)
        # (64,128x128) -> (128,64x64)
        x = self.block2(x)
        # (128,64x64) -> (256,32x32)
        x1 = self.block3(x)
        # (256,32x32) -> (512,16x16)
        x2 = self.block4(x1)

        # (512,16x16) -> (512,8x8)
        x = self.block5(x2)
        # (512,8x8) -> (1024,4x4)
        x = self.block6(x)
        # (1024,4x4) -> (1024,4x4)
        x = self.block7(x)
        x = self.activation(x)
        # (1024,4x4) -> (1024,)
        x = torch.sum(x, dim=(2, 3))
        # (1024,) -> (1,)
        out_im = self.l8(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 128) * ((bbox[:, 4] - bbox[:, 2]) < 128)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        label_l, label_s = label[~s_idx], label[s_idx]

        # (256, 32x32) -> (512, 32x32)
        obj_feat_s = self.block_obj4(x1)
        # (512, 32x32) -> (512, 32x32)
        obj_feat_s = self.block_obj5(obj_feat_s)
        # (512,32x32) -> (cls_s, 512,8x8)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)   # for small object feature, (s, 512, 8, 8)

        # (512,16x16) -> (512,16x16)
        obj_feat_l = self.block_obj5(x2)
        # (512,16x16) -> (cls_l, 512,8x8)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)   # for large object feature, (l, 512, 8, 8)
        # (cls_l, 512,8x8) + (cls_s, 512,8x8) -> (cls, 512,8x8)
        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)  # (cls, 512, 8, 8)
        label = torch.cat([label_l, label_s], dim=0)  # (cls,)

        # (cls, 512,8x8) -> (cls, 1024, 4x4)
        obj_feat = self.block_obj6(obj_feat)
        obj_feat = self.activation(obj_feat)
        # (cls, 1024,4x4) -> (cls, 1024,)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))

        # (cls, 1024,) -> (cls, class_num)
        output_cls = self.classifier_cls(obj_feat)

        # (cls, 1024,) -> (cls, 1,)
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(label).view(cls, -1) * obj_feat.view(cls, -1), dim=1, keepdim=True)

        return out_im, out_obj, output_cls


class RoIStyle(nn.Module):
    def __init__(self, class_num=200, input_dim=512, output_dim=256):
        super(RoIStyle, self).__init__()

        # (512, 8, 8) -> (128, 8, 8)
        self.c0 = nn.Conv2d(input_dim, input_dim // 4, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(input_dim // 4) if class_num == 0 else ConditionalBatchNorm2d(input_dim // 4,
                                                                                                class_num)
        # (128, 8, 8) -> (256, 4, 4)
        self.c1 = nn.Conv2d(input_dim // 4, input_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_dim // 2) if class_num == 0 else ConditionalBatchNorm2d(input_dim // 2,
                                                                                                class_num)
        # # (256, 4, 4) -> (512, 2, 2)
        self.c2 = nn.Conv2d(input_dim // 2, input_dim, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(input_dim) if class_num == 0 else ConditionalBatchNorm2d(input_dim, class_num)
        # pool (512, 2, 2) -> (512, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # (512, 1, 1) -> (256, 1, 1)
        self.fc_mu = nn.Linear(input_dim, output_dim)
        self.fc_logvar = nn.Linear(input_dim, output_dim)

    def forward(self, fm, objs=None):
        x = fm
        x = self.c0(x)
        x = self.bn0(x) if objs is None else self.bn0(x, objs)
        x = self.activation(x)
        x = self.c1(x)
        x = self.bn1(x) if objs is None else self.bn1(x, objs)
        x = self.activation(x)
        x = self.c2(x)
        x = self.bn2(x) if objs is None else self.bn2(x, objs)
        x = self.activation(x)
        x = self.pool(x)

        # resize tensor to a flatten for fully convolution
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = logvar.mul(0.5).exp()
        eps = get_z_random(std.size(0), std.size(1)).to(fm.device)
        z = eps.mul(std).add(mu)

        return z, mu, logvar


def get_z_random(batch_size, z_dim, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, z_dim) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, z_dim)
    return z


class Discriminator(nn.Module):
    def __init__(self, class_num):
        super(Discriminator, self).__init__()

        self.backbone = ResnetRoI(class_num=class_num)

    def forward(self, img, boxes, objs):

        img_score, obj_score, output_cls = self.backbone(img, boxes, objs)

        return img_score, obj_score, output_cls


class Obj_Discriminator(nn.Module):
    def __init__(self, class_num):
        super(Obj_Discriminator, self).__init__()

        self.Obj_backbone_content = Obj_ResnetRoI(class_num=class_num, input_dim=3)

    def forward(self, crops):

        obj_score, output_cls = self.Obj_backbone_content(crops)

        return obj_score, output_cls


class ImageDiscriminator(nn.Module):
    def __init__(self, class_num, embed_dim=256):
        super(ImageDiscriminator, self).__init__()

        self.backbone_content = ResnetRoI(class_num=class_num, input_dim=3, output_dim=c_dim)

        self.roi_style = RoIStyle(class_num=class_num, input_dim=c_dim, output_dim=z_dim)

        self.ch = embed_dim // 16
        self.relu = nn.ReLU()
        self.main = nn.Sequential(
            # (3, 256, 256) -> (16, 128, 128)
            OptimizedBlock(3, self.ch, downsample=True),

            # (16, 128, 128) -> (32, 64, 64)
            ResidualBlock(self.ch, self.ch * 2, downsample=True),
            # (32, 64, 64) -> (64, 32, 32)
            ResidualBlock(self.ch * 2, self.ch * 4, downsample=True),
            # (64, 32, 32) -> (128, 16, 16)
            ResidualBlock(self.ch * 4, self.ch * 8, downsample=True),
            # (128, 16, 16) -> (256, 8, 8)
            ResidualBlock(self.ch * 8, self.ch * 16, downsample=True),

            # (256, 8, 8) -> (512, 4, 4)
            ResidualBlock(self.ch * 16, self.ch * 32, downsample=True),
            # (512, 4, 4) -> (1024, 2, 2)
            ResidualBlock(self.ch * 32, self.ch * 64, downsample=True),
        )
        # (1024,)-> (1,)
        self.classifier = nn.Linear(self.ch * 64, 1, bias=False)

        # self.apply(weights_init)

    def forward(self, x):
        print("ImageDiscriminator: ")
        h = self.main(x)
        h = self.relu(h)
        # print("h.shape: ", h.shape)  # torch.Size([1, 1024, 2, 2])
        # (1024, 2, 2) -> (1024,)
        h = torch.sum(h, dim=(2, 3))
        # print("h.shape: ", h.shape)  # torch.Size([1, 1024])
        output = self.classifier(h)
        # print("output.shape: ", output.shape)  # torch.Size([1, 1])
        return output.view(-1)


class ObjectDiscriminator(nn.Module):
    def __init__(self, conv_dim=64, n_class=133, downsample_first=False):
        super(ObjectDiscriminator, self).__init__()
        self.relu = nn.ReLU()
        self.main = nn.Sequential(
            # (3, 32, 32) -> (64, 32, 32)
            OptimizedBlock(3, conv_dim, downsample=downsample_first),

            # (64, 32, 32) -> (128, 16, 16)
            ResidualBlock(conv_dim, conv_dim * 2, downsample=True),
            # (128, 16, 16) -> (256, 8, 8)
            ResidualBlock(conv_dim * 2, conv_dim * 4, downsample=True),
            # (256, 8, 8) -> (512, 4, 4)
            ResidualBlock(conv_dim * 4, conv_dim * 8, downsample=True),
            # (512, 4, 4) -> (1024, 2, 2)
            ResidualBlock(conv_dim * 8, conv_dim * 16, downsample=True),
        )
        # (1024,) -> (1,)
        self.classifier_src = nn.Linear(conv_dim * 16, 1)
        # (1024,) -> (133,)
        self.classifier_cls = nn.Linear(conv_dim * 16, n_class)

        # if n_class > 0:
        #     self.l_y = nn.Embedding(num_embeddings=n_class, embedding_dim=conv_dim * 16)

        # self.apply(weights_init)

    def forward(self, x, y=None):
        # print("ObjectDiscriminator: ")
        h = x
        h = self.main(h)
        h = self.relu(h)
        print("h.shape: ", h.shape)  # torch.Size([7, 1024, 2, 2])
        # (1024, 2, 2) -> (1024,)
        h = torch.sum(h, dim=(2, 3))
        print("h.shape: ", h.shape)  # torch.Size([7, 1024])

        output_src = self.classifier_src(h)  # (1024,) -> (1,)
        print("output_src.shape: ", output_src.shape)  # torch.Size([7, 1])
        output_cls = self.classifier_cls(h)  # (1024,) -> (133,)
        print("output_cls.shape: ", output_cls.shape)  # torch.Size([7, 133])

        return output_src.view(-1), output_cls


if __name__ == '__main__':
    # from torchsummary import summary
    # from torchsummaryX import summary

    from data.t2v_custom_mask import get_dataloader
    from models.generator import Generator, show_model_parameter

    cudnn.enabled = True
    cudnn.benchmark = True

    # 'cuda:0' for multi-gpu 0, only one gpu just 'cuda'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    embedding_dim = 256
    z_dim = 256
    c_dim = 512
    batch_size = 1
    clstm_layers = 4

    n_res = 4  # number of residual blocks in content encoder/decoder: 4
    activ = 'lrelu'
    pad_type = 'reflect'  # padding type [zero/reflect]: reflect
    mlp_dim = 512

    # 创建训练以及测试得数据迭代器
    dataloader = get_dataloader(batch_size=1, T2V_DIR='../datasets/thermal2visible_256x256_test', is_training=True)

    # print("0 memory allocated in MB:{}".format(torch.cuda.memory_allocated() / 1024 ** 2))  # 0.0

    # using coco panoptic categories.json
    vocab_num = 134

    # 生成网络模型
    netG_t = Generator(class_num=vocab_num, embedding_dim=embedding_dim, z_dim=z_dim, c_dim=c_dim,
                       clstm_layers=clstm_layers).cuda()
    netG_v = Generator(class_num=vocab_num, embedding_dim=embedding_dim, z_dim=z_dim,
                       c_dim=c_dim, clstm_layers=clstm_layers).cuda()

    # 鉴别模型
    netD_image_t = Discriminator(class_num=vocab_num).cuda()
    netD_image_v = Discriminator(class_num=vocab_num).cuda()

    for i, batch in enumerate(dataloader):
        img_t, img_v, objs, boxes, masks, obj_to_img, fname = batch
        style_rand_t = torch.randn(objs.size(0), z_dim)  # Random Norm Distribution and style dim: 256
        style_rand_v = torch.randn(objs.size(0), z_dim)  # Random Norm Distribution and style dim: 256
        img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v = img_t.cuda(), img_v.cuda(), \
               objs.cuda(), boxes.cuda(), masks.cuda(), obj_to_img.cuda(), style_rand_t.cuda(), style_rand_v.cuda()

        # Generator
        fake_image_v = netG_v(img_v, objs, boxes, masks, obj_to_img)
        show_model_parameter(fake_image_v, netG_v, name='image generator')  # total 54041731
        # Discriminator
        img_score_v, obj_score_v, obj_cls_v = netD_image_v(img_v, boxes, objs)
        show_model_parameter(img_score_v, netD_image_v, name='image discriminator')  # total 66751304

        # real img and obj loss:
        d_image_adv_loss_v = torch.nn.ReLU()(1.0 - img_score_v).mean()
        d_object_adv_loss_v = torch.nn.ReLU()(1.0 - obj_score_v).mean()

        # fake img and obj loss:
        img_score_v2v, obj_score_v2v, obj_cls_v2v = netD_image_v(fake_image_v.detach(), boxes, objs)
        d_image_adv_loss_v2v = torch.nn.ReLU()(1.0 + img_score_v2v).mean()
        d_object_adv_loss_v2v = torch.nn.ReLU()(1.0 + obj_score_v2v).mean()

        # object classification loss：
        g_object_cls_loss_v = F.cross_entropy(obj_cls_v, objs)
        g_object_cls_loss_v2v = F.cross_entropy(obj_cls_v2v, objs)


