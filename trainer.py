"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.generator import Generator
from models.discriminator import Discriminator

from utils.model_saver import load_model, save_model
from utils.data import VGGLoss, get_scheduler, weights_init, image_save, html_save
from utils.html import HTML

import torch.nn.functional as F

import torch
import torch.nn as nn
from collections import OrderedDict

import os
import os.path
from pathlib import Path

from imageio import imwrite


class T2V_Trainer(nn.Module):
    def __init__(self, config, local_rank):
        super(T2V_Trainer, self).__init__()

        self.local_rank = local_rank

        # using detectron2 pretrained model
        # category_num = 134, from panoptic 2017 annotation, stuffs: 80 ~ 133, things: 0 ~ 79, so total: 134:
        vocab_num = config.category_num

        assert config.clstm_layers > 0

        # Initiate the networks
        # 生成网络模型t, # auto-encoder for domain thermal
        self.netG_t = Generator(class_num=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim,
                           c_dim=config.c_dim, clstm_layers=config.clstm_layers).cuda()
        # 生成网络模型v, # auto-encoder for domain visible
        self.netG_v = Generator(class_num=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim,
                           c_dim=config.c_dim, clstm_layers=config.clstm_layers).cuda()

        # 鉴别模型a，鉴别生成的图像，是否和数据集A的分布一致:  thermal to visible
        self.netD_image_t = Discriminator(class_num=vocab_num).cuda()
        # 鉴别模型b，鉴别生成的图像，是否和数据集B的分布一致:  visible to thermal
        self.netD_image_v = Discriminator(class_num=vocab_num).cuda()

        # Setup the optimizers， 优化器的超参数
        beta1 = config.beta1
        beta2 = config.beta2

        # 生成模型a,b的相关参数
        gen_params = list(self.netG_t.parameters()) + list(self.netG_v.parameters())
        # 鉴别模型a,b的相关参数
        dis_image_params = list(self.netD_image_t.parameters()) + list(self.netD_image_v.parameters())

        self.netG_optimizer = torch.optim.Adam([p for p in gen_params if p.requires_grad], config.learning_rate_g,
                                               betas=(beta1, beta2))
        self.netD_image_optimizer = torch.optim.Adam([p for p in dis_image_params if p.requires_grad],
                                                     config.learning_rate_d, betas=(beta1, beta2))

        # Reinitilize schedulers 鉴别模型以及生成模型的学习率衰减策略
        self.D_image_scheduler = get_scheduler(self.netD_image_optimizer, config)
        self.G_scheduler = get_scheduler(self.netG_optimizer, config)

        # Network weight initialization
        self.netG_t.apply(weights_init(config.init))
        self.netG_v.apply(weights_init(config.init))
        self.netD_image_t.apply(weights_init(config.init))
        self.netD_image_v.apply(weights_init(config.init))

        # define loss function:
        # self.tv_loss = TVLoss()
        self.vgg_loss = VGGLoss()
        # self.vgg = Vgg16().type(torch.cuda.FloatTensor)
        self.l1_loss = nn.L1Loss()

    def TVLoss(self, image):
        diff_i = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
        diff_j = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
        tv_loss = (diff_i + diff_j) / (256 * 256)
        return tv_loss

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v):
        # 先设定为推断模式
        self.eval()

        # # 1, 把随机噪声转化为pytorch变量
        # style_rand_t = Variable(torch.randn(objs.size(0), config.z_dim).cuda())
        # style_rand_v = Variable(torch.randn(objs.size(0), config.z_dim).cuda())

        # 2, 输入图片t，v进行编码，分别得到content code 以及 style code
        content_t, style_t, _, _ = self.netG_t.encode(img_t, objs, boxes)
        content_v, style_v, _, _ = self.netG_v.encode(img_v, objs, boxes)

        # 3, 对content code 加入 random style code，然后进行解码（混合），得到合成的图片
        img_v2t = self.netG_t.decode(content_v, style_rand_t, objs, masks, obj_to_img)
        img_t2v = self.netG_v.decode(content_t, style_rand_v, objs, masks, obj_to_img)

        self.train()
        return img_v2t, img_t2v

    # 生成模型进行优化
    def gen_update(self, img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v, config, iters,
                   writer, result_save_dir, fname, fname_list):
        # (1) 前向传播
        # Generate fake image
        # 1, Object Style and Content Encoder:
        content_t, style_t, mu_t, logvar_t = self.netG_t.encode(img_t, objs, boxes)
        content_v, style_v, mu_v, logvar_v = self.netG_v.encode(img_v, objs, boxes)

        # 2, decode (within domain)
        img_t2t = self.netG_t.decode(content_t, style_t, objs, boxes, masks, obj_to_img)
        img_v2v = self.netG_v.decode(content_v, style_v, objs, boxes, masks, obj_to_img)

        # 3, decode (cross domain) 进行交叉解码，即两张图片的content code，style code进行互换
        img_v2t = self.netG_t.decode(content_v, style_rand_t, objs, boxes, masks, obj_to_img)
        img_t2v = self.netG_v.decode(content_t, style_rand_v, objs, boxes, masks, obj_to_img)

        # # 4, Identity generating
        # img_t2t_id = self.netG_t.decode(content_t, style_rand_t, objs, boxes, masks, obj_to_img)
        # img_v2v_id = self.netG_v.decode(content_v, style_rand_v, objs, boxes, masks, obj_to_img)

        # 5, encode again，对上面合成的图片再进行编码，得到重构的content code，style code
        content_v_fake, style_t_fake, _, _ = self.netG_t.encode(img_v2t, objs, boxes)
        content_t_fake, style_v_fake, _, _ = self.netG_v.encode(img_t2v, objs, boxes)

        # 6, decode again (if needed)，重构的content code 与真实图片编码得到 style code（s_x_prime）进行解码，生成新图片
        img_t2v2t = self.netG_t.decode(content_t_fake, style_t, objs, boxes, masks, obj_to_img)
        img_v2t2v = self.netG_v.decode(content_v_fake, style_v, objs, boxes, masks, obj_to_img)

        # 7, compute losses
        # (2) 计算损失
        # Image L1 loss:
        g_img_rec_loss_t2t_t = self.l1_loss(img_t2t, img_t).mean()
        g_img_rec_loss_v2v_v = self.l1_loss(img_v2v, img_v).mean()
        g_img_rec_loss_t2v2t_t = self.l1_loss(img_t2v2t, img_t).mean()
        g_img_rec_loss_v2t2v_v = self.l1_loss(img_v2t2v, img_v).mean()
        g_img_L1_loss = g_img_rec_loss_t2t_t + g_img_rec_loss_v2v_v \
                        + g_img_rec_loss_t2v2t_t + g_img_rec_loss_v2t2v_v

        # # Identity loss
        # g_identity_loss_t2t_id_t = self.l1_loss(img_t2t_id, img_t).mean()
        # g_identity_loss_v2v_id_v = self.l1_loss(img_v2v_id, img_v).mean()
        # g_identity_loss = g_identity_loss_t2t_id_t + g_identity_loss_v2v_id_v

        # Object style loss
        g_z_fake_loss_v2t_t = self.l1_loss(style_t_fake, style_rand_t).mean()
        g_z_fake_loss_t2v_v = self.l1_loss(style_v_fake, style_rand_v).mean()
        g_fake_z_L1_loss = g_z_fake_loss_v2t_t + g_z_fake_loss_t2v_v

        # Object  content loss
        g_c_fake_loss_v2t_v = self.l1_loss(content_v_fake, content_v).mean()
        g_c_fake_loss_t2v_t = self.l1_loss(content_t_fake, content_t).mean()
        g_fake_c_L1_loss = g_c_fake_loss_v2t_v + g_c_fake_loss_t2v_t

        # fake img and obj adversal loss:
        g_img_score_v2t, g_obj_score_v2t, g_obj_cls_v2t = self.netD_image_t(img_v2t, boxes, objs)
        g_loss_fake_v2t = - g_img_score_v2t.mean()
        g_loss_obj_v2t = - g_obj_score_v2t.mean()
        g_img_score_t2v, g_obj_score_t2v, g_obj_cls_t2v = self.netD_image_v(img_t2v, boxes, objs)
        g_loss_fake_t2v = - g_img_score_t2v.mean()
        g_loss_obj_t2v = - g_obj_score_t2v.mean()
        g_img_score_t2t, g_obj_score_t2t, g_obj_cls_t2t = self.netD_image_t(img_t2t, boxes, objs)
        g_loss_fake_t2t = - g_img_score_t2t.mean()
        g_loss_obj_t2t = - g_obj_score_t2t.mean()
        g_img_score_v2v, g_obj_score_v2v, g_obj_cls_v2v = self.netD_image_v(img_v2v, boxes, objs)
        g_loss_fake_v2v = - g_img_score_v2v.mean()
        g_loss_obj_v2v = - g_obj_score_v2v.mean()
        # g_img_score_t2t_id, g_obj_score_t2t_id, g_obj_cls_t2t_id = self.netD_image_t(img_t2t_id, boxes, objs)
        # g_loss_fake_t2t_id = - g_img_score_t2t_id.mean()
        # g_loss_obj_t2t_id = - g_obj_score_t2t_id.mean()
        # g_img_score_v2v_id, g_obj_score_v2v_id, g_obj_cls_v2v_id = self.netD_image_v(img_v2v_id, boxes, objs)
        # g_loss_fake_v2v_id = - g_img_score_v2v_id.mean()
        # g_loss_obj_v2v_id = - g_obj_score_v2v_id.mean()
        # g_img_score_t2v2t, g_obj_score_t2v2t, g_obj_cls_t2v2t = self.netD_image_t(img_t2v2t, boxes, objs)
        # g_loss_fake_t2v2t = - g_img_score_t2v2t.mean()
        # g_loss_obj_t2v2t = - g_obj_score_t2v2t.mean()
        # g_img_score_v2t2v, g_obj_score_v2t2v, g_obj_cls_v2t2v = self.netD_image_v(img_v2t2v, boxes, objs)
        # g_loss_fake_v2t2v = - g_img_score_v2t2v.mean()
        # g_loss_obj_v2t2v = - g_obj_score_v2t2v.mean()
        g_img_adv_loss = g_loss_fake_v2t + g_loss_fake_t2v \
                         + g_loss_fake_t2t + g_loss_fake_v2v
                         # + g_loss_fake_t2v2t + g_loss_fake_v2t2v
                         # + g_loss_fake_t2t_id + g_loss_fake_v2v_id

        g_obj_adv_loss = g_loss_obj_v2t + g_loss_obj_t2v \
                         + g_loss_obj_t2t + g_loss_obj_v2v
                         # + g_loss_obj_t2v2t + g_loss_obj_v2t2v
                         # + g_loss_obj_t2t_id + g_loss_obj_v2v_id

        # fake object classification loss：
        g_obj_cls_loss_v2t = F.cross_entropy(g_obj_cls_v2t, objs)
        g_obj_cls_loss_t2v = F.cross_entropy(g_obj_cls_t2v, objs)
        g_obj_cls_loss_t2t = F.cross_entropy(g_obj_cls_t2t, objs)
        g_obj_cls_loss_v2v = F.cross_entropy(g_obj_cls_v2v, objs)
        # g_obj_cls_loss_t2t_id = F.cross_entropy(g_obj_cls_t2t_id, objs)
        # g_obj_cls_loss_v2v_id = F.cross_entropy(g_obj_cls_v2v_id, objs)
        # g_obj_cls_loss_t2v2t = F.cross_entropy(g_obj_cls_t2v2t, objs)
        # g_obj_cls_loss_v2t2v = F.cross_entropy(g_obj_cls_v2t2v, objs)
        g_obj_cls_loss = g_obj_cls_loss_v2t + g_obj_cls_loss_t2v \
                         + g_obj_cls_loss_t2t + g_obj_cls_loss_v2v
                         # + g_obj_cls_loss_t2v2t + g_obj_cls_loss_v2t2v
                         # + g_obj_cls_loss_t2t_id + g_obj_cls_loss_v2v_id

        # perceptual loss:
        perceptual_loss_t2t = self.vgg_loss(img_t2t, img_t).mean()
        perceptual_loss_v2v = self.vgg_loss(img_v2v, img_v).mean()
        # perceptual_loss_t2t_id = self.vgg_loss(img_t2t_id, img_t).mean()
        # perceptual_loss_v2v_id = self.vgg_loss(img_v2v_id, img_v).mean()
        perceptual_loss_t2v = self.vgg_loss(img_t2v, img_t).mean()
        perceptual_loss_v2t = self.vgg_loss(img_v2t, img_v).mean()
        perceptual_loss_t2v2t_t = self.vgg_loss(img_t2v2t, img_t).mean()
        perceptual_loss_v2t2v_v = self.vgg_loss(img_v2t2v, img_v).mean()
        g_perceptual_loss = perceptual_loss_t2t + perceptual_loss_v2v \
                            + perceptual_loss_t2v + perceptual_loss_v2t \
                            + perceptual_loss_t2v2t_t + perceptual_loss_v2t2v_v
                            # + perceptual_loss_t2t_id + perceptual_loss_v2v_id \

        # tv loss: The total variation (TV) loss encourages spatial smoothness in the generated image
        tv_loss_t2v = self.TVLoss(img_t2v)
        tv_loss_v2t = self.TVLoss(img_v2t)
        tv_loss_t2t = self.TVLoss(img_t2t)
        tv_loss_v2v = self.TVLoss(img_v2v)
        # tv_loss_t2t_id = self.TVLoss(img_t2t_id)
        # tv_loss_v2v_id = self.TVLoss(img_v2v_id)
        tv_loss_t2v2t = self.TVLoss(img_t2v2t)
        tv_loss_v2t2v = self.TVLoss(img_v2t2v)
        g_tv_loss = tv_loss_t2v + tv_loss_v2t \
                    + tv_loss_t2t + tv_loss_v2v \
                    + tv_loss_t2v2t + tv_loss_v2t2v
                    # + tv_loss_t2t_id + tv_loss_v2v_id

        # kl loss:
        kl_element_t = mu_t.pow(2).add_(logvar_t.exp()).mul_(-1).add_(1).add_(logvar_t)
        g_kl_loss_t = torch.sum(kl_element_t).mul_(-0.5)
        kl_element_v = mu_v.pow(2).add_(logvar_v.exp()).mul_(-1).add_(1).add_(logvar_v)
        g_kl_loss_v = torch.sum(kl_element_v).mul_(-0.5)
        g_kl_loss = g_kl_loss_t + g_kl_loss_v

        # Backward and optimize.
        g_loss = 0
        g_loss += config.lambda_img_rec * g_img_L1_loss
        # g_loss += config.lambda_identity * g_identity_loss
        g_loss += config.lambda_z_rec * g_fake_z_L1_loss
        g_loss += config.lambda_c_rec * g_fake_c_L1_loss
        g_loss += config.lambda_img_adv * g_img_adv_loss
        g_loss += config.lambda_obj_adv * g_obj_adv_loss
        g_loss += config.lambda_obj_cls * g_obj_cls_loss
        g_loss += config.lambda_kl * g_kl_loss
        g_loss += config.lambda_vgg * g_perceptual_loss
        g_loss += config.lambda_tv * g_tv_loss

        # ######## 梯度累加trick: 实现低显存跑大batchsize ########
        # # 损失标准化
        # g_loss = g_loss / config.accumulation_steps
        # # 计算梯度
        # g_loss.backward()
        # if (iters + 1) % config.accumulation_steps == 0:
        #     # 反向传播，更新网络参数
        #     self.netG_optimizer.step()
        #
        #     # 更新学习率
        #     if self.G_scheduler is not None:
        #         self.G_scheduler.step()
        #
        #     # 清空梯度
        #     self.netG_t.zero_grad()
        #     self.netG_v.zero_grad()
        #
        #     # Logging.
        #     loss = {}
        #     loss['G/loss'] = g_loss.item()
        #     loss['G/fake_image_adv_loss'] = g_img_adv_loss.item()
        #     loss['G/fake_object_adv_loss'] = g_obj_adv_loss.item()
        #     loss['G/fake_object_cls_loss'] = g_obj_cls_loss.item()
        #     loss['G/image_L1_loss'] = g_img_L1_loss.item()
        #     loss['G/object_style_loss'] = g_fake_z_L1_loss.item()
        #     loss['G/object_content_loss'] = g_fake_c_L1_loss.item()
        #     loss['G/kl_loss'] = g_kl_loss.item()
        #     loss['G/perceptual_loss'] = g_perceptual_loss.item()
        #     loss['G/tv_loss'] = g_tv_loss.item()
        #     loss['G/identity_loss'] = g_identity_loss.item()
        # ######## 梯度累加trick: 实现低显存跑大batchsize ########

        # (3) 清空梯度
        self.netG_t.zero_grad()
        self.netG_v.zero_grad()

        # (4) 计算梯度
        g_loss.backward()

        # (5) 反向传播， 更新网络参数
        self.netG_optimizer.step()

        # 更新学习率
        if self.G_scheduler is not None:
            self.G_scheduler.step()

        # Logging.
        loss = {}
        loss['G/loss'] = g_loss.item()
        loss['G/fake_image_adv_loss'] = g_img_adv_loss.item()
        loss['G/fake_object_adv_loss'] = g_obj_adv_loss.item()
        loss['G/fake_object_cls_loss'] = g_obj_cls_loss.item()
        loss['G/image_L1_loss'] = g_img_L1_loss.item()
        loss['G/object_style_loss'] = g_fake_z_L1_loss.item()
        loss['G/object_content_loss'] = g_fake_c_L1_loss.item()
        loss['G/kl_loss'] = g_kl_loss.item()
        loss['G/perceptual_loss'] = g_perceptual_loss.item()
        loss['G/tv_loss'] = g_tv_loss.item()
        # loss['G/identity_loss'] = g_identity_loss.item()

        if (iters + 1) % config.log_step == 0:
            if self.local_rank == 0:
                log = 'iter [{:06d}/{:06d}]'.format(iters + 1, config.niter)
                for tag, roi_value in loss.items():
                    log += ", {}: {:.4f}".format(tag, roi_value)
                print(log)

        if (iters + 1) % config.tensorboard_step == 0 and config.use_tensorboard:
            if self.local_rank == 0:
                for tag, roi_value in loss.items():
                    writer.add_scalar(tag, roi_value, iters + 1)

        if (iters + 1) % config.image_step == 0 and config.image_step:
            if self.local_rank == 0:
                # Real Thermal:
                image_save('img_%08d_t_%s' % (iters + 1, fname), img_t, result_save_dir + "/images")
                # Real Visible:
                image_save('img_%08d_v_%s' % (iters + 1, fname), img_v, result_save_dir + "/images")
                # Thermal to Visible:
                image_save('img_%08d_t2v_%s' % (iters + 1, fname), img_t2v, result_save_dir + "/images")
                # write to html
                html_save(iters + 1, result_save_dir, config, fname_list)

    # 鉴别模型进行优化
    def dis_update(self, img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v, config, iters, writer):
        # (1) 前向传播
        # Generate fake image
        # 1, Object Style and Content Encoder:
        content_t, style_t, mu_t, logvar_t = self.netG_t.encode(img_t, objs, boxes)
        content_v, style_v, mu_v, logvar_v = self.netG_v.encode(img_v, objs, boxes)

        # 2, decode (within domain)
        img_t2t = self.netG_t.decode(content_t, style_t, objs, boxes, masks, obj_to_img)
        img_v2v = self.netG_v.decode(content_v, style_v, objs, boxes, masks, obj_to_img)

        # 3, decode (cross domain) 进行交叉解码，即两张图片的content code，style code进行互换
        img_v2t = self.netG_t.decode(content_v, style_rand_t, objs, boxes, masks, obj_to_img)
        img_t2v = self.netG_v.decode(content_t, style_rand_v, objs, boxes, masks, obj_to_img)

        # # 4, Identity generating
        # img_t2t_id = self.netG_t.decode(content_t, style_rand_t, objs, boxes, masks, obj_to_img)
        # img_v2v_id = self.netG_v.decode(content_v, style_rand_v, objs, boxes, masks, obj_to_img)

        # 5, encode again，对上面合成的图片再进行编码，得到重构的content code，style code
        content_v_fake, style_t_fake, _, _ = self.netG_t.encode(img_v2t, objs, boxes)
        content_t_fake, style_v_fake, _, _ = self.netG_v.encode(img_t2v, objs, boxes)

        # 6, decode again (if needed)，重构的content code 与真实图片编码得到 style code（s_x_prime）进行解码，生成新图片
        img_t2v2t = self.netG_t.decode(content_t_fake, style_t, objs, boxes, masks, obj_to_img)
        img_v2t2v = self.netG_v.decode(content_v_fake, style_v, objs, boxes, masks, obj_to_img)

        # 7, compute losses
        # (2) 计算损失
        # real img and obj loss:
        img_score_t, obj_score_t, obj_cls_t = self.netD_image_t(img_t, boxes, objs)
        d_image_adv_loss_t = torch.nn.ReLU()(1.0 - img_score_t).mean()
        d_object_adv_loss_t = torch.nn.ReLU()(1.0 - obj_score_t).mean()
        img_score_v, obj_score_v, obj_cls_v = self.netD_image_v(img_v, boxes, objs)
        d_image_adv_loss_v = torch.nn.ReLU()(1.0 - img_score_v).mean()
        d_object_adv_loss_v = torch.nn.ReLU()(1.0 - obj_score_v).mean()
        d_real_image_adv_loss = d_image_adv_loss_t + d_image_adv_loss_v
        d_real_object_adv_loss = d_object_adv_loss_t + d_object_adv_loss_v

        # fake img and obj loss:
        img_score_v2t, obj_score_v2t, _ = self.netD_image_t(img_v2t.detach(), boxes, objs)
        d_image_adv_loss_v2t = torch.nn.ReLU()(1.0 + img_score_v2t).mean()
        d_object_adv_loss_v2t = torch.nn.ReLU()(1.0 + obj_score_v2t).mean()
        img_score_t2v, obj_score_t2v, _ = self.netD_image_v(img_t2v.detach(), boxes, objs)
        d_image_adv_loss_t2v = torch.nn.ReLU()(1.0 + img_score_t2v).mean()
        d_object_adv_loss_t2v = torch.nn.ReLU()(1.0 + obj_score_t2v).mean()
        img_score_t2t, obj_score_t2t, _ = self.netD_image_t(img_t2t.detach(), boxes, objs)
        d_image_adv_loss_t2t = torch.nn.ReLU()(1.0 + img_score_t2t).mean()
        d_object_adv_loss_t2t = torch.nn.ReLU()(1.0 + obj_score_t2t).mean()
        img_score_v2v, obj_score_v2v, _ = self.netD_image_v(img_v2v.detach(), boxes, objs)
        d_image_adv_loss_v2v = torch.nn.ReLU()(1.0 + img_score_v2v).mean()
        d_object_adv_loss_v2v = torch.nn.ReLU()(1.0 + obj_score_v2v).mean()
        # img_score_t2t_id, obj_score_t2t_id, _ = self.netD_image_t(img_t2t_id.detach(), boxes, objs)
        # d_image_adv_loss_t2t_id = torch.nn.ReLU()(1.0 + img_score_t2t_id).mean()
        # d_object_adv_loss_t2t_id = torch.nn.ReLU()(1.0 + obj_score_t2t_id).mean()
        # img_score_v2v_id, obj_score_v2v_id, _ = self.netD_image_v(img_v2v_id.detach(), boxes, objs)
        # d_image_adv_loss_v2v_id = torch.nn.ReLU()(1.0 + img_score_v2v_id).mean()
        # d_object_adv_loss_v2v_id = torch.nn.ReLU()(1.0 + obj_score_v2v_id).mean()
        # img_score_t2v2t, obj_score_t2v2t, _ = self.netD_image_t(img_t2v2t.detach(), boxes, objs)
        # d_image_adv_loss_t2v2t = torch.nn.ReLU()(1.0 + img_score_t2v2t).mean()
        # d_object_adv_loss_t2v2t = torch.nn.ReLU()(1.0 + obj_score_t2v2t).mean()
        # img_score_v2t2v, obj_score_v2t2v, _ = self.netD_image_v(img_v2t2v.detach(), boxes, objs)
        # d_image_adv_loss_v2t2v = torch.nn.ReLU()(1.0 + img_score_v2t2v).mean()
        # d_object_adv_loss_v2t2v = torch.nn.ReLU()(1.0 + obj_score_v2t2v).mean()
        d_fake_image_adv_loss = d_image_adv_loss_v2t + d_image_adv_loss_t2v \
                                + d_image_adv_loss_t2t + d_image_adv_loss_v2v
                                # + d_image_adv_loss_t2v2t + d_image_adv_loss_v2t2v
                                # + d_image_adv_loss_t2t_id + d_image_adv_loss_v2v_id

        d_fake_object_adv_loss = d_object_adv_loss_v2t + d_object_adv_loss_t2v \
                                 + d_object_adv_loss_t2t + d_object_adv_loss_v2v
                                 # + d_object_adv_loss_t2v2t + d_object_adv_loss_v2t2v
                                 # + d_object_adv_loss_t2t_id + d_object_adv_loss_v2v_id

        # real object classification loss：
        d_object_cls_loss_t = F.cross_entropy(obj_cls_t, objs)
        d_object_cls_loss_v = F.cross_entropy(obj_cls_v, objs)
        d_real_object_cls_loss = d_object_cls_loss_t + d_object_cls_loss_v

        # Backward and optimize.
        d_loss = 0
        d_loss += config.lambda_img_adv * (d_fake_image_adv_loss + d_real_image_adv_loss)
        d_loss += config.lambda_obj_adv * (d_fake_object_adv_loss + d_real_object_adv_loss)
        d_loss += config.lambda_obj_cls * d_real_object_cls_loss

        # ######## 梯度累加trick: 实现低显存跑大batchsize ########
        # # 损失标准化
        # d_loss = d_loss / config.accumulation_steps
        # # 计算梯度
        # d_loss.backward()
        # if (iters + 1) % config.accumulation_steps == 0:
        #     # 反向传播，更新网络参数
        #     self.netD_image_optimizer.step()
        #
        #     # 更新学习率
        #     if self.D_image_scheduler is not None:
        #         self.D_image_scheduler.step()
        #
        #     # 清空梯度
        #     self.netD_image_t.zero_grad()
        #     self.netD_image_v.zero_grad()
        #
        #     # Logging.
        #     loss = {}
        #     loss['D/loss'] = d_loss.item()
        #     loss['D/real_image_adv_loss'] = d_real_image_adv_loss.item()
        #     loss['D/fake_image_adv_loss'] = d_fake_image_adv_loss.item()
        #     loss['D/real_object_adv_loss'] = d_real_object_adv_loss.item()
        #     loss['D/fake_object_adv_loss'] = d_fake_object_adv_loss.item()
        #     loss['D/fake_object_cls_loss'] = d_real_object_cls_loss.item()
        # ######## 梯度累加trick: 实现低显存跑大batchsize ########

        # (3) 清空梯度
        self.netD_image_t.zero_grad()
        self.netD_image_v.zero_grad()

        # (4) 计算梯度
        d_loss.backward()

        # (5) 反向传播， 更新网络参数
        self.netD_image_optimizer.step()

        # 更新学习率
        if self.D_image_scheduler is not None:
            self.D_image_scheduler.step()

        # Logging.
        loss = {}
        loss['D/loss'] = d_loss.item()
        loss['D/real_image_adv_loss'] = d_real_image_adv_loss.item()
        loss['D/fake_image_adv_loss'] = d_fake_image_adv_loss.item()
        loss['D/real_object_adv_loss'] = d_real_object_adv_loss.item()
        loss['D/fake_object_adv_loss'] = d_fake_object_adv_loss.item()
        loss['D/fake_object_cls_loss'] = d_real_object_cls_loss.item()

        if (iters + 1) % config.log_step == 0:
            if self.local_rank == 0:
                log = 'iter [{:06d}/{:06d}]'.format(iters + 1, config.niter)
                for tag, roi_value in loss.items():
                    log += ", {}: {:.4f}".format(tag, roi_value)
                print(log)

        if (iters + 1) % config.tensorboard_step == 0 and config.use_tensorboard:
            if self.local_rank == 0:
                for tag, roi_value in loss.items():
                    # add_scalars是将不同变量添加到同一个图下，图的名称是add_scalars第一个变量
                    writer.add_scalar(tag, roi_value, iters + 1)

    def resume(self, model_save_dir, config, map_location):
        # Load models
        start_iter = load_model(self.netG_t, model_dir=model_save_dir, appendix='netG_t', iter=config.resume_iter,
                                  map_location=map_location)
        _ = load_model(self.netG_v, model_dir=model_save_dir, appendix='netG_v', iter=config.resume_iter,
                                  map_location=map_location)
        _ = load_model(self.netD_image_t, model_dir=model_save_dir, appendix='netD_image_t', iter=config.resume_iter,
                       map_location=map_location)
        _ = load_model(self.netD_image_v, model_dir=model_save_dir, appendix='netD_image_v', iter=config.resume_iter,
                       map_location=map_location)
        # Load optimizers
        _ = load_model(self.netG_optimizer, model_dir=model_save_dir, appendix='netG_optimizer', iter=config.resume_iter,
                       map_location=map_location)
        _ = load_model(self.netD_image_optimizer, model_dir=model_save_dir, appendix='netD_image_optimizer', iter=config.resume_iter,
                       map_location=map_location)
        # Reinitilize schedulers 鉴别模型以及生成模型的学习率衰减策略
        self.D_image_scheduler = get_scheduler(self.netD_image_optimizer, config, start_iter)
        self.G_scheduler = get_scheduler(self.netG_optimizer, config, start_iter)

        return start_iter

    def save(self, model_save_dir, iters, config):
        # Save generators, discriminators, and optimizers
        save_model(self.netG_t, model_dir=model_save_dir, appendix='netG_t', iter=iters + 1, save_num=5,
                   save_step=config.save_step)
        save_model(self.netG_v, model_dir=model_save_dir, appendix='netG_v', iter=iters + 1, save_num=5,
                   save_step=config.save_step)
        # save_model(self.netD_image_t, model_dir=model_save_dir, appendix='netD_image_t', iter=iters + 1,
        #            save_num=5, save_step=config.save_step)
        # save_model(self.netD_image_v, model_dir=model_save_dir, appendix='netD_image_v', iter=iters + 1,
        #            save_num=5, save_step=config.save_step)
        # save_model(self.netG_optimizer, model_dir=model_save_dir, appendix='netG_optimizer', iter=iters + 1,
        #            save_num=5, save_step=config.save_step)
        # save_model(self.netD_image_optimizer, model_dir=model_save_dir, appendix='netD_image_optimizer',
        #            iter=iters + 1, save_num=5, save_step=config.save_step)
