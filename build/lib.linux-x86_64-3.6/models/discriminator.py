import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import json
import numpy as np
from models.layers import build_cnn
from models.layers import GlobalAvgPool
from models.bilinear import crop_bbox_batch
# from models.initialization import weights_init
import torch.backends.cudnn as cudnn

from utils.data import init_weights


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


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _downsample(x):
    return F.avg_pool2d(x, kernel_size=2)


class OptimizedBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.downsample = downsample

        self.resi = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
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


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class GANLoss_multi(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss_multi, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator_multi(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'multi':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D=3, getIntermFeat=False)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator_multi, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_multi(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class ImageDiscriminator(nn.Module):
    def __init__(self, embed_dim=256):
        super(ImageDiscriminator, self).__init__()
        self.ch = embed_dim // 16
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
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
        h = self.lrelu(h)
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
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
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
        h = self.lrelu(h)
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
    from torchsummaryX import summary

    from data.t2v_custom_mask import get_dataloader

    from models.generator import Generator, show_model_parameter

    import random

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
    activ = 'lrelu'  # activation function [relu/lrelu/prelu/selu/tanh]: relu
    pad_type = 'reflect'  # padding type [zero/reflect]: reflect
    mlp_dim = 512  # number of filters in MLP: 32

    # 创建训练以及测试得数据迭代器
    loader_t, loader_v = get_dataloader(batch_size=1, T2V_DIR='../datasets/thermal2visible_256x256', is_training=True)

    # print("0 memory allocated in MB:{}".format(torch.cuda.memory_allocated() / 1024 ** 2))  # 0.0

    # using coco panoptic categories.json
    vocab_num = 133

    # # 生成网络模型, # auto-encoder
    # netG_t = Generator(num_embeddings=vocab_num, embedding_dim=embedding_dim, z_dim=z_dim, c_dim=c_dim,
    #                    clstm_layers=clstm_layers).cuda()
    netG_v = Generator(num_embeddings=vocab_num, embedding_dim=embedding_dim, z_dim=z_dim,
                       c_dim=c_dim, clstm_layers=clstm_layers).cuda()

    # # 鉴别模型a，鉴别生成的图像，是否和数据集A的分布一致:  thermal to visible
    # netD_image_t = ImageDiscriminator(embed_dim=embedding_dim).cuda()
    # netD_object_t = ObjectDiscriminator(n_class=vocab_num).cuda()

    # 鉴别模型b，鉴别生成的图像，是否和数据集B的分布一致:  visible to thermal
    netD_image_v = ImageDiscriminator(embed_dim=embedding_dim).cuda()
    netD_object_v = ObjectDiscriminator(n_class=vocab_num).cuda()

    # netD_image_t = add_sn(netD_image_t)
    # netD_object_t = add_sn(netD_object_t)
    netD_image_v = add_sn(netD_image_v)
    netD_object_v = add_sn(netD_object_v)

    for i, (batch_t, batch_v) in enumerate(zip(loader_t, loader_v)):
        img_t, objs, boxes, masks, obj_to_img = batch_t
        img_v, _, _, _, _ = batch_v
        style_rand_t = torch.randn(objs.size(0), z_dim).cuda()
        style_rand_v = torch.randn(objs.size(0), z_dim).cuda()
        img_t, img_v, objs, boxes, masks, obj_to_img = img_t.cuda(), img_v.cuda(), objs.cuda(), boxes.cuda(), \
                                                       masks.cuda(), obj_to_img.cuda()

        # test Generator
        fake_image_v = netG_v(img_v, objs, boxes, masks, obj_to_img)
        fake_crops_v = crop_bbox_batch(fake_image_v, boxes, obj_to_img, obj_size)
        # print("fake_image_v.shape: ", fake_image_v.shape)  # torch.Size([1, 3, 256, 256])
        # print("fake_crops_v.shape: ", fake_crops_v.shape)  # torch.Size([cls, 3, 32, 32])

        out_logits_v = netD_image_v(fake_image_v)  # torch.Size([1,])
        # show_model_parameter(out_logits_v, netD_image_v, name='discriminator_image')
        # total 32-256d:10142656

        g_image_adv_loss_v = F.binary_cross_entropy_with_logits(out_logits_v,
                                                                  torch.full_like(out_logits_v,
                                                                                  random.uniform(0.9, 1)))
        # print("image_adv_v.shape: ", g_image_adv_loss_v.shape)  # torch.Size([])
        # print("image_adv_v: ", g_image_adv_loss_v)  # tensor(0.6360, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)

        out_logits_src_v, out_logits_cls_v = netD_object_v(fake_crops_v, objs)  # torch.Size([cls,]), torch.Size([cls,133])
        # show_model_parameter(out_logits_src_v, netD_object_v, name='discriminator_object')
        # show_model_parameter(out_logits_cls_v, netD_object_v, name='discriminator_object')
        # total 32-256d:10277766

        g_object_adv_loss_rand_v = F.binary_cross_entropy_with_logits(out_logits_src_v,
                                                                        torch.full_like(out_logits_src_v,
                                                                                        random.uniform(0.9, 1)))
        # print("object_adv_v.shape: ", g_object_adv_loss_rand_v.shape)  # torch.Size([])
        # print("object_adv_v: ", g_object_adv_loss_rand_v)  # tensor(0.6437, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)

        g_object_cls_loss_rand_v = F.cross_entropy(out_logits_cls_v, objs)
        # print("object_cls_v: ", g_object_cls_loss_rand_v.shape)  # torch.Size([])
        # print("object_cls_v: ", g_object_cls_loss_rand_v)  # tensor(4.7615, device='cuda:0', grad_fn=<PermuteBackward>)


