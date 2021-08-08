import torch
import argparse

from trainer import T2V_Trainer

# 代码里导入tensorboardX
from tensorboardX import SummaryWriter

from data.t2v_custom_mask import get_dataloader as get_dataloader_t2v

from utils.model_saver import prepare_dir
from utils.miscs import str2bool

import torch.backends.cudnn as cudnn

import sys
import time

from torch.nn.parallel import DistributedDataParallel


def main(config):
    # 可以提升一点训练速度，没有额外开销，一般都会加, also can avoid GPU memory adding
    cudnn.enabled = True
    cudnn.benchmark = True

    # print(config.local_rank)
    local_rank = config.local_rank

    # 初始化使用nccl后端
    torch.distributed.init_process_group(backend="nccl")

    # device = torch.device("cuda:{}".format(local_rank))
    torch.cuda.set_device(local_rank)

    # Setup model and data loader, 根据配置创建模型
    if config.trainer == 'T2V':
        trainer = T2V_Trainer(config, local_rank).cuda()
    else:
        sys.exit("Only support T2V|?")

    # # 同步BN，也就是SyncBN，只在DataDistributedParallel（DDP）中支持。可以实现真正的多卡BN。引入SyncBN，这会将普通BN替换成SyncBN。
    # trainer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(trainer)

    # Initiate trainer of the networks
    trainer = DistributedDataParallel(trainer, device_ids=[local_rank], output_device=local_rank)

    # save data
    gen_log_save_dir, dis_log_save_dir, model_save_dir, result_save_dir = prepare_dir(config.dir, config.exp_name)

    # 创建训练以及测试得数据迭代器
    # capture dataset:
    train_loader = get_dataloader_t2v(batch_size=config.batch_size, T2V_DIR=config.t2v_dir,
                                      is_training=config.is_training)

    # # using coco panoptic categories.json
    # vocab_num = 134
    #
    # assert config.clstm_layers > 0
    #
    # # Initiate the networks
    # # 生成网络模型t, # auto-encoder for domain thermal
    # netG_t = Generator(num_embeddings=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim,
    #                    c_dim=config.c_dim, clstm_layers=config.clstm_layers).cuda()
    # # 生成网络模型v, # auto-encoder for domain visible
    # netG_v = Generator(num_embeddings=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim,
    #                    c_dim=config.c_dim, clstm_layers=config.clstm_layers).cuda()
    #
    # # method 1: DataParallel for mlti-gpu training, but its loads are not balanced for each gpu!
    # # netG_t = nn.DataParallel(netG_t)
    # # netG_v = nn.DataParallel(netG_v)
    #
    # # method 2: DataParallelModel for mlti-gpu training with good load balanced, but outputs of model are separated
    # # and not tensor but list/tuple!
    # # netG_t = DataParallelModel(netG_t)
    # # netG_v = DataParallelModel(netG_v)
    #
    # # method 3: DistributedDataParallel for distributed mlti-gpu training, dataset processing and computation with
    # # good load balanced (recommended!)
    # # netG_t = DistributedDataParallel(netG_t, device_ids=[local_rank], output_device=local_rank)
    # # netG_v = DistributedDataParallel(netG_v, device_ids=[local_rank], output_device=local_rank)
    #
    # # 鉴别模型a，鉴别生成的图像，是否和数据集A的分布一致:  thermal to visible
    # netD_image_t = ImageDiscriminator(embed_dim=config.embedding_dim).cuda()
    # netD_object_t = ObjectDiscriminator(n_class=vocab_num).cuda()
    #
    # # 鉴别模型b，鉴别生成的图像，是否和数据集B的分布一致:  visible to thermal
    # netD_image_v = ImageDiscriminator(embed_dim=config.embedding_dim).cuda()
    # netD_object_v = ObjectDiscriminator(n_class=vocab_num).cuda()
    #
    # # netD_image_t = nn.DataParallel(netD_image_t)
    # # netD_image_v = nn.DataParallel(netD_image_v)
    # # netD_object_t = nn.DataParallel(netD_object_t)
    # # netD_object_v = nn.DataParallel(netD_object_v)
    #
    # # netD_image_t = DataParallelModel(netD_image_t)
    # # netD_image_v = DataParallelModel(netD_image_v)
    # # netD_object_t = DataParallelModel(netD_object_t)
    # # netD_object_v = DataParallelModel(netD_object_v)
    #
    # # netD_image_t = DistributedDataParallel(netD_image_t, device_ids=[local_rank], output_device=local_rank)
    # # netD_object_t = DistributedDataParallel(netD_object_t, device_ids=[local_rank], output_device=local_rank)
    # # netD_image_v = DistributedDataParallel(netD_image_v, device_ids=[local_rank], output_device=local_rank)
    # # netD_object_v = DistributedDataParallel(netD_object_v, device_ids=[local_rank], output_device=local_rank)
    #
    # netD_image_t = add_sn(netD_image_t)
    # netD_object_t = add_sn(netD_object_t)
    # netD_image_v = add_sn(netD_image_v)
    # netD_object_v = add_sn(netD_object_v)
    #
    # # Setup the optimizers， 优化器的超参数
    # beta1 = config.beta1
    # beta2 = config.beta2
    #
    # # 生成模型a,b的相关参数
    # gen_params = list(netG_t.parameters()) + list(netG_v.parameters())
    # # 鉴别模型a,b的相关参数
    # dis_params = list(netD_image_t.parameters()) + list(netD_image_v.parameters())
    # # 鉴别模型a,b object的相关参数
    # dis_object_params = list(netD_object_t.parameters()) + list(netD_object_v.parameters())
    #
    # netG_optimizer = torch.optim.Adam([p for p in gen_params if p.requires_grad], config.learning_rate,
    #                                   betas=(beta1, beta2), weight_decay=config.weight_decay)
    # netD_image_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad], config.learning_rate,
    #                                         betas=(beta1, beta2), weight_decay=config.weight_decay)
    # netD_object_optimizer = torch.optim.Adam([p for p in dis_object_params if p.requires_grad], config.learning_rate,
    #                                          betas=(beta1, beta2), weight_decay=config.weight_decay)
    #
    # # Reinitilize schedulers 鉴别模型以及生成模型的学习率衰减策略
    # D_image_scheduler = get_scheduler(netD_image_optimizer, config)
    # D_object_scheduler = get_scheduler(netD_object_optimizer, config)
    # G_scheduler = get_scheduler(netG_optimizer, config)
    #
    # # # Network weight initialization: Discriminator，网络模型权重初始化
    # netD_image_t.apply(weights_init(config.init))
    # netD_object_t.apply(weights_init(config.init))
    # netD_image_v.apply(weights_init(config.init))
    # netD_object_v.apply(weights_init(config.init))
    #
    # # # Load VGG model if needed，加载VGG模型，用来计算感知 loss
    # # if config.vgg_w > 0:
    # #     pretrained_models_dir = os.path.join(config.dir, 'checkpoints/pretrained/models')
    # #     if not Path(pretrained_models_dir).exists():
    # #         Path(pretrained_models_dir).mkdir(parents=True)
    # #     # distributed model loading way:
    # #     map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    # #     vgg = load_vgg16(pretrained_models_dir, map_location=map_location)
    # #     vgg.eval()
    # #     for param in vgg.parameters():
    # #         param.requires_grad = False
    #
    # # We only save the model which uses device "cuda:0"
    # # To resume, the device for the saved model would also be "cuda:0"
    # if config.resume:
    #     map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    #     # ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))
    #     # Load models and optimizers:
    #     start_iter = load_model(netG_t, model_dir=model_save_dir, appendix='netG_t', iter=config.resume_iter,
    #                             map_location=map_location)
    #     _ = load_model(netD_image_t, model_dir=model_save_dir, appendix='netD_image_t', iter=config.resume_iter,
    #                    map_location=map_location)
    #     _ = load_model(netD_object_t, model_dir=model_save_dir, appendix='netD_object_t', iter=config.resume_iter,
    #                    map_location=map_location)
    #     _ = load_model(netG_v, model_dir=model_save_dir, appendix='netG_v', iter=config.resume_iter,
    #                    map_location=map_location)
    #     _ = load_model(netD_image_v, model_dir=model_save_dir, appendix='netD_image_v', iter=config.resume_iter,
    #                    map_location=map_location)
    #     _ = load_model(netD_object_v, model_dir=model_save_dir, appendix='netD_object_v', iter=config.resume_iter,
    #                    map_location=map_location)
    #     _ = load_model(netG_optimizer, model_dir=model_save_dir, appendix='netG_optimizer', iter=config.resume_iter,
    #                    map_location=map_location)
    #     _ = load_model(netD_image_optimizer, model_dir=model_save_dir, appendix='netD_image_optimizer',
    #                    iter=config.resume_iter, map_location=map_location)
    #     _ = load_model(netD_object_optimizer, model_dir=model_save_dir, appendix='netD_object_optimizer',
    #                    iter=config.resume_iter, map_location=map_location)
    #     # Reinitilize schedulers 鉴别模型以及生成模型的学习率衰减策略
    #     D_image_scheduler = get_scheduler(netD_image_optimizer, config, start_iter)
    #     D_object_scheduler = get_scheduler(netD_object_optimizer, config, start_iter)
    #     G_scheduler = get_scheduler(netG_optimizer, config, start_iter)
    # print("model loaded!")

    # Start training，开始训练模型，如果设置opts.resume=Ture,表示接着之前得训练
    if config.resume:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        start_iter = trainer.module.resume(model_save_dir, config=config, map_location=map_location)

    data_iter = iter(train_loader)

    if start_iter < config.niter:

        print("GPU: {}, Training Start from epoch: {} ...".format(local_rank, start_iter))

        start_time = time.time()

        if config.use_tensorboard:
            # 将loss写到 log_save_dir 路径下面
            # writer = SummaryWriter(log_save_dir)
            writer_discriminator = SummaryWriter(dis_log_save_dir)
            writer_generator = SummaryWriter(gen_log_save_dir)
            # print("SummaryWriter is runninng...")

        fname_list = []

        for i in range(start_iter, config.niter):
            try:
                batch = next(data_iter)
            except:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # print("memory allocated in MB:{}".format(torch.cuda.memory_allocated() / 1024 ** 2))
            # result here -> dataset loaded memory allocated in MB: 363.44384765625 -> MB: 1112.5673828125

            # print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, i))

            # 指定数据存储计算的设备
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            img_t, img_v, objs, boxes, masks, obj_to_img, fname = batch
            style_rand_t = torch.randn(objs.size(0), config.z_dim)  # Random Norm Distribution and style dim: 256
            style_rand_v = torch.randn(objs.size(0), config.z_dim)  # Random Norm Distribution and style dim: 256
            img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v = img_t.cuda(), img_v.cuda(), \
                   objs.cuda(), boxes.cuda(), masks.cuda(), obj_to_img.cuda(), style_rand_t.cuda(), style_rand_v.cuda()

            # with Timer("Elapsed time in update: %f"):
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
            # if (i == 0) or ((i + 1) % config.discriminator_steps == 0):
            trainer.module.dis_update(img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v, config,
                                      i, writer_discriminator)

            # # (1) 前向传播
            # # Generate fake image
            # # 2, Object Style and Content Encoder:
            # crops_t, content_t, style_t, _, _ = netG_t.module.encode(img_t, objs, boxes, obj_to_img)
            # crops_v, content_v, style_v, _, _ = netG_v.module.encode(img_v, objs, boxes, obj_to_img)
            #
            # # 3, decode (within domain)
            # img_t2t = netG_t.module.decode(content_t, style_t, objs, masks, obj_to_img)
            # img_v2v = netG_v.module.decode(content_v, style_v, objs, masks, obj_to_img)
            #
            # # 4, decode (cross domain) 进行交叉解码，即两张图片的content code，style code进行互换
            # img_v2t = netG_t.module.decode(content_v, style_rand_t, objs, masks, obj_to_img)
            # img_t2v = netG_v.module.decode(content_t, style_rand_v, objs, masks, obj_to_img)
            #
            # # 5, encode again，对上面合成的图片再进行编码，得到重构的content code，style code
            # crops_v2t, content_v_fake, style_crops_v2t, _, _ = netG_t.module.encode(img_v2t, objs, boxes, obj_to_img)
            # crops_t2v, content_t_fake, style_crops_t2v, _, _ = netG_v.module.encode(img_t2v, objs, boxes, obj_to_img)
            # crops_t2t, _, _, _, _ = netG_t.module.encode(img_t2t, objs, boxes, obj_to_img)
            # crops_v2v, _, _, _, _ = netG_v.module.encode(img_v2v, objs, boxes, obj_to_img)
            #
            # # 6, decode again (if needed)，重构的content code 与真实图片编码得到 style code（s_x_prime）进行解码，生成新图片
            # img_t2v2t = netG_t.module.decode(content_t_fake, style_t, objs, masks, obj_to_img)
            # img_v2t2v = netG_v.module.decode(content_v_fake, style_v, objs, masks, obj_to_img)
            # crops_t2v2t, _, _, _, _ = netG_t.module.encode(img_t2v2t, objs, boxes, obj_to_img)
            # crops_v2t2v, _, _, _, _ = netG_v.module.encode(img_v2t2v, objs, boxes, obj_to_img)
            #
            # # (2) 计算损失
            # # 1, fake image adv loss: fake images != real image -> 0
            # logits_t2t = netD_image_t(img_t2t.detach())
            # d_image_adv_loss_t2t = F.binary_cross_entropy_with_logits(logits_t2t,
            #                                                           torch.full_like(logits_t2t,
            #                                                                           random.uniform(0, 0.1)))
            # logits_v2v = netD_image_v(img_v2v.detach())
            # d_image_adv_loss_v2v = F.binary_cross_entropy_with_logits(logits_v2v,
            #                                                           torch.full_like(logits_v2v,
            #                                                                           random.uniform(0, 0.1)))
            # logits_v2t = netD_image_t(img_v2t.detach())
            # d_image_adv_loss_v2t = F.binary_cross_entropy_with_logits(logits_v2t,
            #                                                           torch.full_like(logits_v2t,
            #                                                                           random.uniform(0, 0.1)))
            # logits_t2v = netD_image_v(img_t2v.detach())
            # d_image_adv_loss_t2v = F.binary_cross_entropy_with_logits(logits_t2v,
            #                                                           torch.full_like(logits_t2v,
            #                                                                           random.uniform(0, 0.1)))
            # logits_t2v2t = netD_image_t(img_t2v2t.detach())
            # d_image_adv_loss_t2v2t = F.binary_cross_entropy_with_logits(logits_t2v2t,
            #                                                             torch.full_like(logits_t2v2t,
            #                                                                             random.uniform(0, 0.1)))
            # logits_v2t2v = netD_image_v(img_v2t2v.detach())
            # d_image_adv_loss_v2t2v = F.binary_cross_entropy_with_logits(logits_v2t2v,
            #                                                             torch.full_like(logits_v2t2v,
            #                                                                             random.uniform(0, 0.1)))
            # d_fake_image_adv_loss = d_image_adv_loss_t2t + d_image_adv_loss_v2v + d_image_adv_loss_v2t + \
            #                         d_image_adv_loss_t2v + d_image_adv_loss_t2v2t + d_image_adv_loss_v2t2v
            #
            # # 2, input image adv loss: input images = real images -> 1
            # logits_t = netD_image_t(img_t)
            # d_image_adv_loss_t = F.binary_cross_entropy_with_logits(logits_t,
            #                                                         torch.full_like(logits_t, random.uniform(0.9, 1)))
            # logits_v = netD_image_v(img_v)
            # d_image_adv_loss_v = F.binary_cross_entropy_with_logits(logits_v,
            #                                                         torch.full_like(logits_v, random.uniform(0.9, 1)))
            # d_real_image_adv_loss = d_image_adv_loss_t + d_image_adv_loss_v
            #
            # # 3, fake image crops adv loss: fake image objects != real image objects -> 0
            # logits_crops_t2t, _ = netD_object_t(crops_t2t.detach(), objs)
            # d_object_adv_loss_crops_t2t = F.binary_cross_entropy_with_logits(logits_crops_t2t,
            #                                                                  torch.full_like(logits_crops_t2t,
            #                                                                                  random.uniform(0, 0.1)))
            # logits_crops_v2v, _ = netD_object_v(crops_v2v.detach(), objs)
            # d_object_adv_loss_crops_v2v = F.binary_cross_entropy_with_logits(logits_crops_v2v,
            #                                                                  torch.full_like(logits_crops_v2v,
            #                                                                                  random.uniform(0, 0.1)))
            # logits_crops_v2t, _ = netD_object_t(crops_v2t.detach(), objs)
            # d_object_adv_loss_crops_v2t = F.binary_cross_entropy_with_logits(logits_crops_v2t,
            #                                                                  torch.full_like(logits_crops_v2t,
            #                                                                                  random.uniform(0, 0.1)))
            # logits_crops_t2v, _ = netD_object_v(crops_t2v.detach(), objs)
            # d_object_adv_loss_crops_t2v = F.binary_cross_entropy_with_logits(logits_crops_t2v,
            #                                                                  torch.full_like(logits_crops_t2v,
            #                                                                                  random.uniform(0, 0.1)))
            # logits_crops_t2v2t, _ = netD_object_t(crops_t2v2t.detach(), objs)
            # d_object_adv_loss_crops_t2v2t = F.binary_cross_entropy_with_logits(logits_crops_t2v2t,
            #                                                                    torch.full_like(logits_crops_t2v2t,
            #                                                                                    random.uniform(0, 0.1)))
            # logits_crops_v2t2v, _ = netD_object_v(crops_v2t2v.detach(), objs)
            # d_object_adv_loss_crops_v2t2v = F.binary_cross_entropy_with_logits(logits_crops_v2t2v,
            #                                                                    torch.full_like(logits_crops_v2t2v,
            #                                                                                    random.uniform(0, 0.1)))
            # d_fake_object_adv_loss = d_object_adv_loss_crops_t2t + d_object_adv_loss_crops_v2v + \
            #                          d_object_adv_loss_crops_v2t + d_object_adv_loss_crops_t2v + \
            #                          d_object_adv_loss_crops_t2v2t + d_object_adv_loss_crops_v2t2v
            #
            # # 4, input image crops adv loss: input image objects = real image objects -> 1
            # logits_crops_t, logits_cls_crops_t = netD_object_t(crops_t.detach(), objs)
            # d_object_adv_loss_t = F.binary_cross_entropy_with_logits(logits_crops_t,
            #                                                          torch.full_like(logits_crops_t,
            #                                                                          random.uniform(0.9, 1)))
            # logits_crops_v, logits_cls_crops_v = netD_object_v(crops_v.detach(), objs)
            # d_object_adv_loss_v = F.binary_cross_entropy_with_logits(logits_crops_v, torch.full_like(logits_crops_v,
            #                                                                                          random.uniform(0.9,
            #                                                                                                         1)))
            # d_real_object_adv_loss = d_object_adv_loss_t + d_object_adv_loss_v
            #
            # # 5, input image crops classification loss: input image objects classification -> 1
            # d_object_cls_loss_real_t = F.cross_entropy(logits_cls_crops_t, objs)
            # d_object_cls_loss_real_v = F.cross_entropy(logits_cls_crops_v, objs)
            # d_real_object_cls_loss = d_object_cls_loss_real_t + d_object_cls_loss_real_v
            #
            # # Backward and optimize. D loss 计算鉴别器的loss
            # d_loss = 0
            # d_image_loss = 0
            # d_object_loss = 0
            # d_image_loss += config.lambda_img_adv * (d_fake_image_adv_loss + d_real_image_adv_loss)
            # d_object_loss += config.lambda_obj_adv * (d_fake_object_adv_loss + d_real_object_adv_loss)
            # d_object_loss += config.lambda_obj_cls * d_real_object_cls_loss
            # d_loss += d_image_loss + d_object_loss
            #
            # # ######## 梯度累加trick: 实现低显存跑大batchsize ########
            # # # 损失标准化
            # # d_loss = d_loss / config.accumulation_steps
            # # # 计算梯度
            # # d_loss.backward()
            # # if (i + 1) % config.accumulation_steps == 0:
            # #     # 反向传播，更新网络参数
            # #     netD_image_optimizer.step()
            # #     netD_object_optimizer.step()
            # #
            # #     # 更新学习率
            # #     if D_image_scheduler is not None:
            # #         D_image_scheduler.step()
            # #     if D_object_scheduler is not None:
            # #         D_object_scheduler.step()
            # #
            # #     # 清空梯度
            # #     netD_image_t.zero_grad()
            # #     netD_object_t.zero_grad()
            # #     netD_image_v.zero_grad()
            # #     netD_object_v.zero_grad()
            # #
            # #     loss = {}
            # #     loss['D/loss'] = d_loss.item()
            # #     loss['D/real_image_adv_loss'] = d_real_image_adv_loss.item()
            # #     loss['D/fake_image_adv_loss'] = d_fake_image_adv_loss.item()
            # #     loss['D/real_object_adv_loss'] = d_real_object_adv_loss.item()
            # #     loss['D/fake_object_adv_loss'] = d_fake_object_adv_loss.item()
            # #     loss['D/real_object_cls_loss'] = d_real_object_cls_loss.item()
            # # ######## 梯度累加trick: 实现低显存跑大batchsize ########
            #
            # # (3) 清空梯度
            # # 当optimizer = optim.Optimizer(net.parameters())时，model.zero_grad()和optimizer.zero_grad()二者等效
            # netD_image_t.zero_grad()
            # netD_object_t.zero_grad()
            # netD_image_v.zero_grad()
            # netD_object_v.zero_grad()
            #
            # # (4) 计算梯度
            # d_loss.backward()
            #
            # # (5) 反向传播， 更新网络参数
            # netD_image_optimizer.step()
            # netD_object_optimizer.step()
            #
            # # 更新学习率
            # if D_image_scheduler is not None:
            #     D_image_scheduler.step()
            # if D_object_scheduler is not None:
            #     D_object_scheduler.step()
            #
            # # 对不同GPU进程上的Loss取平均, 在进行数据记录:
            # d_loss = reduce_tensor(d_loss.data)
            # d_real_image_adv_loss = reduce_tensor(d_real_image_adv_loss.data)
            # d_fake_image_adv_loss = reduce_tensor(d_fake_image_adv_loss.data)
            # d_real_object_adv_loss = reduce_tensor(d_real_object_adv_loss.data)
            # d_fake_object_adv_loss = reduce_tensor(d_fake_object_adv_loss.data)
            # d_real_object_cls_loss = reduce_tensor(d_real_object_cls_loss.data)
            #
            # # Logging. loss.item can avoid GPU memory adding by training time
            # loss = {}
            # loss['D/d_loss'] = d_loss.item()
            # loss['D/real_image_adv_loss'] = d_real_image_adv_loss.item()
            # loss['D/fake_image_adv_loss'] = d_fake_image_adv_loss.item()
            # loss['D/real_object_adv_loss'] = d_real_object_adv_loss.item()
            # loss['D/fake_object_adv_loss'] = d_fake_object_adv_loss.item()
            # loss['D/real_object_cls_loss'] = d_real_object_cls_loss.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            fname_list.append(fname)
            trainer.module.gen_update(img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v,
                                          config, i, writer_generator, result_save_dir, fname, fname_list)
            # torch.cuda.synchronize()  # just used for calculate time of each iteration

            # # Generate fake image
            # # 2, Object Style and Content Encoder:
            # crops_t, content_t, style_t, mu_t, logvar_t = netG_t.module.encode(img_t, objs, boxes, obj_to_img)
            # crops_v, content_v, style_v, mu_v, logvar_v = netG_v.module.encode(img_v, objs, boxes, obj_to_img)
            #
            # # 3, decode (within domain)
            # img_t2t = netG_t.module.decode(content_t, style_t, objs, masks, obj_to_img)
            # img_v2v = netG_v.module.decode(content_v, style_v, objs, masks, obj_to_img)
            #
            # # 4, decode (cross domain) 进行交叉解码，即两张图片的content code，style code进行互换
            # img_v2t = netG_t.module.decode(content_v, style_rand_t, objs, masks, obj_to_img)
            # img_t2v = netG_v.module.decode(content_t, style_rand_v, objs, masks, obj_to_img)
            #
            # # 5, encode again，对上面合成的图片再进行编码，得到重构的content code，style code
            # crops_v2t, content_v_fake, style_crops_v2t, _, _ = netG_t.module.encode(img_v2t, objs, boxes, obj_to_img)
            # crops_t2v, content_t_fake, style_crops_t2v, _, _ = netG_v.module.encode(img_t2v, objs, boxes, obj_to_img)
            # crops_t2t, _, _, _, _ = netG_t.module.encode(img_t2t, objs, boxes, obj_to_img)
            # crops_v2v, _, _, _, _ = netG_v.module.encode(img_v2v, objs, boxes, obj_to_img)
            #
            # # 6, decode again (if needed)，重构的content code 与真实图片编码得到 style code（s_x_prime）进行解码，生成新图片
            # img_t2v2t = netG_t.module.decode(content_t_fake, style_t, objs, masks, obj_to_img)
            # img_v2t2v = netG_v.module.decode(content_v_fake, style_v, objs, masks, obj_to_img)
            # crops_t2v2t, _, _, _, _ = netG_t.module.encode(img_t2v2t, objs, boxes, obj_to_img)
            # crops_v2t2v, _, _, _, _ = netG_v.module.encode(img_v2t2v, objs, boxes, obj_to_img)
            #
            # # 7, compute losses
            # # (2) 计算损失
            # # 1), L1 loss (reconstruction loss): reconstructed images = input image
            # g_img_rec_loss_t2t_t = torch.abs(img_t2t - img_t).mean()
            # g_img_rec_loss_v2v_v = torch.abs(img_v2v - img_v).mean()
            # g_rec_img_L1_loss = g_img_rec_loss_t2t_t + g_img_rec_loss_v2v_v
            #
            # g_img_rec_loss_t2v2t_t = torch.abs(img_t2v2t - img_t).mean()
            # g_img_rec_loss_v2t2v_v = torch.abs(img_v2t2v - img_v).mean()
            # g_cycle_img_L1_loss = g_img_rec_loss_t2v2t_t + g_img_rec_loss_v2t2v_v
            #
            # # 2), L1 loss (object latent style code loss): fake image objects style = input random style code
            # g_z_fake_loss_v2t_t = torch.abs(style_crops_v2t - style_rand_t).mean()
            # g_z_fake_loss_t2v_v = torch.abs(style_crops_t2v - style_rand_v).mean()
            # g_fake_z_L1_loss = g_z_fake_loss_v2t_t + g_z_fake_loss_t2v_v
            #
            # # 3), L1 loss (object latent content code loss): fake image objects content = input crops content code
            # g_c_fake_loss_v2t_v = torch.abs(content_v_fake - content_v).mean()
            # g_c_fake_loss_t2v_t = torch.abs(content_t_fake - content_t).mean()
            # g_fake_c_L1_loss = g_c_fake_loss_v2t_v + g_c_fake_loss_t2v_t
            #
            # # 4), kl loss: (KL 散度: 全称叫kullback leibler 散度 or 相对熵, 用来衡量一个分布和另一个分布的相似程度.
            # # encoder输出的 mu [latent mean] 和 logvar [latent log variance] 向量表示的正态分布和标准正态分布间的相似程度，相似程度越高损失函数的值就会越小。
            # # Here it make  style code by encoder tends to a norm distribution so that the rand code is similar to it
            # kl_element_t = mu_t.pow(2).add_(logvar_t.exp()).mul_(-1).add_(1).add_(logvar_t)
            # g_kl_loss_t = torch.sum(kl_element_t).mul_(-0.5)
            # kl_element_v = mu_v.pow(2).add_(logvar_v.exp()).mul_(-1).add_(1).add_(logvar_v)
            # g_kl_loss_v = torch.sum(kl_element_v).mul_(-0.5)
            # g_kl_loss = g_kl_loss_t + g_kl_loss_v
            #
            # # 5), fake image adv loss: fake images = real image -> 1
            # out_logits_t2t = netD_image_t(img_t2t)
            # g_image_adv_loss_t2t = F.binary_cross_entropy_with_logits(out_logits_t2t,
            #                                                           torch.full_like(out_logits_t2t,
            #                                                                           random.uniform(0.9, 1)))
            # out_logits_v2v = netD_image_v(img_v2v)
            # g_image_adv_loss_v2v = F.binary_cross_entropy_with_logits(out_logits_v2v,
            #                                                           torch.full_like(out_logits_v2v,
            #                                                                           random.uniform(0.9, 1)))
            # out_logits_v2t = netD_image_t(img_v2t)
            # g_image_adv_loss_v2t = F.binary_cross_entropy_with_logits(out_logits_v2t,
            #                                                           torch.full_like(out_logits_v2t,
            #                                                                           random.uniform(0.9, 1)))
            # out_logits_t2v = netD_image_v(img_t2v)
            # g_image_adv_loss_t2v = F.binary_cross_entropy_with_logits(out_logits_t2v,
            #                                                           torch.full_like(out_logits_t2v,
            #                                                                           random.uniform(0.9, 1)))
            # out_logits_t2v2t = netD_image_t(img_t2v2t)
            # g_image_adv_loss_t2v2t = F.binary_cross_entropy_with_logits(out_logits_t2v2t,
            #                                                             torch.full_like(out_logits_t2v2t,
            #                                                                             random.uniform(0.9, 1)))
            # out_logits_v2t2v = netD_image_v(img_v2t2v)
            # g_image_adv_loss_v2t2v = F.binary_cross_entropy_with_logits(out_logits_v2t2v,
            #                                                             torch.full_like(out_logits_v2t2v,
            #                                                                             random.uniform(0.9, 1)))
            # g_fake_image_adv_loss = g_image_adv_loss_t2t + g_image_adv_loss_v2v + g_image_adv_loss_v2t + \
            #                         g_image_adv_loss_t2v + g_image_adv_loss_t2v2t + g_image_adv_loss_v2t2v
            #
            # # 6), image crops adv loss: fake image objects = real image objects -> 1
            # out_logits_src_t2t, out_logits_cls_t2t = netD_object_t(crops_t2t, objs)
            # g_object_adv_loss_rec_t2t = F.binary_cross_entropy_with_logits(out_logits_src_t2t,
            #                                                                torch.full_like(out_logits_src_t2t,
            #                                                                                random.uniform(0.9, 1)))
            # out_logits_src_v2v, out_logits_cls_v2v = netD_object_v(crops_v2v, objs)
            # g_object_adv_loss_rec_v2v = F.binary_cross_entropy_with_logits(out_logits_src_v2v,
            #                                                                torch.full_like(out_logits_src_v2v,
            #                                                                                random.uniform(0.9, 1)))
            # out_logits_src_v2t, out_logits_cls_v2t = netD_object_t(crops_v2t, objs)
            # g_object_adv_loss_rand_v2t = F.binary_cross_entropy_with_logits(out_logits_src_v2t,
            #                                                                 torch.full_like(out_logits_src_v2t,
            #                                                                                 random.uniform(0.9, 1)))
            # out_logits_src_t2v, out_logits_cls_t2v = netD_object_v(crops_t2v, objs)
            # g_object_adv_loss_rand_t2v = F.binary_cross_entropy_with_logits(out_logits_src_t2v,
            #                                                                 torch.full_like(out_logits_src_t2v,
            #                                                                                 random.uniform(0.9, 1)))
            # out_logits_src_t2v2t, out_logits_cls_t2v2t = netD_object_t(crops_t2v2t, objs)
            # g_object_adv_loss_rand_t2v2t = F.binary_cross_entropy_with_logits(out_logits_src_t2v2t,
            #                                                                   torch.full_like(out_logits_src_t2v2t,
            #                                                                                   random.uniform(0.9, 1)))
            # out_logits_src_v2t2v, out_logits_cls_v2t2v = netD_object_v(crops_v2t2v, objs)
            # g_object_adv_loss_rand_v2t2v = F.binary_cross_entropy_with_logits(out_logits_src_v2t2v,
            #                                                                   torch.full_like(out_logits_src_v2t2v,
            #                                                                                   random.uniform(0.9, 1)))
            # g_fake_object_adv_loss = g_object_adv_loss_rec_t2t + g_object_adv_loss_rec_v2v + g_object_adv_loss_rand_v2t + \
            #                          g_object_adv_loss_rand_t2v + g_object_adv_loss_rand_t2v2t + g_object_adv_loss_rand_v2t2v
            #
            # # 7), crops classification loss: Auxiliary Classification for each cropped objects with their class
            # g_object_cls_loss_rec_t2t = F.cross_entropy(out_logits_cls_t2t, objs)
            # g_object_cls_loss_rec_v2v = F.cross_entropy(out_logits_cls_v2v, objs)
            # g_object_cls_loss_rand_v2t = F.cross_entropy(out_logits_cls_v2t, objs)
            # g_object_cls_loss_rand_t2v = F.cross_entropy(out_logits_cls_t2v, objs)
            # g_object_cls_loss_rand_t2v2t = F.cross_entropy(out_logits_cls_t2v2t, objs)
            # g_object_cls_loss_rand_v2t2v = F.cross_entropy(out_logits_cls_v2t2v, objs)
            # g_fake_object_cls_loss = g_object_cls_loss_rec_t2t + g_object_cls_loss_rec_v2v + g_object_cls_loss_rand_v2t + \
            #                          g_object_cls_loss_rand_t2v + g_object_cls_loss_rand_t2v2t + g_object_cls_loss_rand_v2t2v
            #
            # # # 8), perceptual loss by vgg: domain-invariant 使用VGG计算感知loss
            # # if config.vgg_w > 0:
            # #     g_image_vgg_loss_t = compute_vgg_loss(vgg, img_v2t, img_v)
            # #     g_image_vgg_loss_v = compute_vgg_loss(vgg, img_t2v, img_t)
            # #     g_object_vgg_loss_t = compute_vgg_loss(vgg, crops_v2t, crops_v)
            # #     g_object_vgg_loss_v = compute_vgg_loss(vgg, crops_t2v, crops_t)
            # #     g_perceptual_loss = g_image_vgg_loss_t + g_image_vgg_loss_v + g_object_vgg_loss_t + g_object_vgg_loss_v
            # # else:
            # #     g_perceptual_loss = 0
            #
            # # Backward and optimize.
            # g_loss = 0
            # g_loss += config.lambda_img_rec * g_rec_img_L1_loss
            # g_loss += config.lambda_img_cyc * g_cycle_img_L1_loss
            # g_loss += config.lambda_z_rec * g_fake_z_L1_loss
            # g_loss += config.lambda_c_rec * g_fake_c_L1_loss
            # g_loss += config.lambda_img_adv * g_fake_image_adv_loss
            # g_loss += config.lambda_obj_adv * g_fake_object_adv_loss
            # g_loss += config.lambda_obj_cls * g_fake_object_cls_loss
            # g_loss += config.lambda_kl * g_kl_loss
            # # g_loss += config.vgg_w * g_perceptual_loss
            #
            # # ######## 梯度累加trick: 实现低显存跑大batchsize ########
            # # # 损失标准化
            # # g_loss = g_loss / config.accumulation_steps
            # # # 计算梯度
            # # g_loss.backward()
            # # if (i + 1) % config.accumulation_steps == 0:
            # #     # 反向传播，更新网络参数
            # #     netG_optimizer.step()
            # #
            # #     # 更新学习率
            # #     if G_scheduler is not None:
            # #         G_scheduler.step()
            # #
            # #     # 清空梯度
            # #     netG_t.zero_grad()
            # #     netG_v.zero_grad()
            # #
            # #     # Logging.
            # #     # loss = {}
            # #     loss['G/loss'] = g_loss.item()
            # #     loss['G/fake_image_adv_loss'] = g_fake_image_adv_loss.item()
            # #     loss['G/fake_object_adv_loss'] = g_fake_object_adv_loss.item()
            # #     loss['G/fake_object_cls_loss'] = g_fake_object_cls_loss.item()
            # #     loss['G/rec_img_L1_loss'] = g_rec_img_L1_loss.item()
            # #     loss['G/cycle_img_L1_loss'] = g_cycle_img_L1_loss.item()
            # #     loss['G/fake_z_L1_loss'] = g_fake_z_L1_loss.item()
            # #     loss['G/fake_c_L1_loss'] = g_fake_c_L1_loss.item()
            # #     loss['G/kl_loss'] = g_kl_loss.item()
            # #     # loss['G/perceptual_loss'] = g_perceptual_loss.item()
            # # ######## 梯度累加trick: 实现低显存跑大batchsize ########
            #
            # # (3) 清空梯度
            # netG_t.zero_grad()
            # netG_v.zero_grad()
            #
            # # (4) 计算梯度
            # g_loss.backward()
            #
            # # (5) 反向传播， 更新网络参数
            # netG_optimizer.step()
            #
            # # 更新学习率
            # if G_scheduler is not None:
            #     G_scheduler.step()
            #
            # # 对不同GPU进程上的Loss取平均, 在进行数据记录:
            # g_loss = reduce_tensor(g_loss.data)
            # g_fake_image_adv_loss = reduce_tensor(g_fake_image_adv_loss.data)
            # g_fake_object_adv_loss = reduce_tensor(g_fake_object_adv_loss.data)
            # g_fake_object_cls_loss = reduce_tensor(g_fake_object_cls_loss.data)
            # g_rec_img_L1_loss = reduce_tensor(g_rec_img_L1_loss.data)
            # g_cycle_img_L1_loss = reduce_tensor(g_cycle_img_L1_loss.data)
            # g_fake_z_L1_loss = reduce_tensor(g_fake_z_L1_loss.data)
            # g_fake_c_L1_loss = reduce_tensor(g_fake_c_L1_loss.data)
            # g_kl_loss = reduce_tensor(g_kl_loss.data)
            #
            # # Logging.
            # loss['G/loss'] = g_loss.item()
            # loss['G/fake_image_adv_loss'] = g_fake_image_adv_loss.item()
            # loss['G/fake_object_adv_loss'] = g_fake_object_adv_loss.item()
            # loss['G/fake_object_cls_loss'] = g_fake_object_cls_loss.item()
            # loss['G/rec_img_L1_loss'] = g_rec_img_L1_loss.item()
            # loss['G/cycle_img_L1_loss'] = g_cycle_img_L1_loss.item()
            # loss['G/fake_z_L1_loss'] = g_fake_z_L1_loss.item()
            # loss['G/fake_c_L1_loss'] = g_fake_c_L1_loss.item()
            # loss['G/kl_loss'] = g_kl_loss.item()
            # # loss['G/perceptual_loss'] = g_perceptual_loss.item()

            # =================================================================================== #
            #                               4. Log                                                #
            # =================================================================================== #

            # # 注意在写入TensorBoard的时候只让一个进程写入就够了：
            # if (i + 1) % config.log_step == 0:
            #     if local_rank == 0:
            #         log = 'iter [{:06d}/{:06d}]'.format(i + 1, config.niter)
            #         for tag, roi_value in loss.items():
            #             log += ", {}: {:.4f}".format(tag, roi_value)
            #         print(log)
            #
            # if (i + 1) % config.tensorboard_step == 0 and config.use_tensorboard:
            #     if local_rank == 0:
            #         for tag, roi_value in loss.items():
            #             # add_scalars是将不同变量添加到同一个图下，图的名称是add_scalars第一个变量
            #             writer.add_scalar(tag, roi_value, i + 1)
            #         # Real Thermal:
            #         img_save('img_t', img_t, i, result_save_dir)
            #         # Real Visible:
            #         img_save('img_v', img_v, i, result_save_dir)
            #         # Thermal to Visible:
            #         img_save('img_t2v', img_t2v, i, result_save_dir)
            #         # Visible to Thermal:
            #         img_save('img_v2t', img_v2t, i, result_save_dir)

            if (i + 1) % config.save_step == 0:
                if local_rank == 0:
                    trainer.module.save(model_save_dir, i, config)
                    # # Save generators, discriminators, and optimizers
                    # save_model(netG_t, model_dir=model_save_dir, appendix='netG_t', iter=i + 1, save_num=5,
                    #            save_step=config.save_step)
                    # save_model(netD_image_t, model_dir=model_save_dir, appendix='netD_image_t', iter=i + 1,
                    #            save_num=5, save_step=config.save_step)
                    # save_model(netD_object_t, model_dir=model_save_dir, appendix='netD_object_t', iter=i + 1,
                    #            save_num=5, save_step=config.save_step)
                    # save_model(netG_v, model_dir=model_save_dir, appendix='netG_v', iter=i + 1, save_num=5,
                    #            save_step=config.save_step)
                    # save_model(netD_image_v, model_dir=model_save_dir, appendix='netD_image_v', iter=i + 1,
                    #            save_num=5, save_step=config.save_step)
                    # save_model(netD_object_v, model_dir=model_save_dir, appendix='netD_object_v', iter=i + 1,
                    #            save_num=5, save_step=config.save_step)
                    # save_model(netG_optimizer, model_dir=model_save_dir, appendix='netG_optimizer', iter=i + 1,
                    #            save_num=5, save_step=config.save_step)
                    # save_model(netD_image_optimizer, model_dir=model_save_dir, appendix='netD_image_optimizer',
                    #            iter=i + 1, save_num=5, save_step=config.save_step)
                    # save_model(netD_object_optimizer, model_dir=model_save_dir, appendix='netD_object_optimizer',
                    #            iter=i + 1, save_num=5, save_step=config.save_step)

            # 如果超过最大迭代次数，则退出训练
            if (i + 1) >= config.niter:
                sys.exit('Finished training!')

        if config.use_tensorboard:
            # writer.close()
            writer_discriminator.close()
            writer_generator.close()

    # print the total time took:
    print("Elapsed total time of {} epochs is: {} hours".format(config.niter, (time.time() - start_time) / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration
    parser.add_argument('--dataset', type=str, default='thermal2visible_256x256', help='dataset selection')
    parser.add_argument('--vg_dir', type=str, default='../datasets/vg')
    parser.add_argument('--coco_dir', type=str, default='../datasets/coco')
    parser.add_argument('--dir', type=str, default='/tmp/OL-GANs')
    parser.add_argument('--t2v_dir', type=str, default='/tmp/OL-GANs/datasets/thermal2visible_256x256_aug')
    parser.add_argument('--is_training', type=str2bool, default='true')
    parser.add_argument('--batch_size', type=int, default=1, help='select batch size')
    parser.add_argument('--niter', type=int, default=1000000, help='number of training iteration')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--object_size', type=int, default=32, help='object size')
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--category_num', type=int, default=134, help='stuffs: 80 ~ 133, things: 0 ~ 79, so total: 134')
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--c_dim', type=int, default=512)
    parser.add_argument('--learning_rate_d', type=float, default=0.005)
    parser.add_argument('--learning_rate_g', type=float, default=1e-4)
    parser.add_argument('--lr_policy', type=str, default='step', help="learning rate scheduler")
    parser.add_argument('--gamma', type=float, default=0.5, help="how much to decay learning rate")
    # because loss update using accumulation gradient (4 x iters) so lr also will update after 4 x step_size iters
    parser.add_argument('--step_size', type=int, default=100000, help="how often to decay learning rate")
    parser.add_argument('--clstm_layers', type=int, default=3)

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--beta1', type=int, default=0.0, help='Adam parameter')
    parser.add_argument('--beta2', type=int, default=0.999, help='Adam parameter')

    # Loss weight
    parser.add_argument('--lambda_img_adv', type=float, default=1.0, help='weight of adv img')
    parser.add_argument('--lambda_obj_adv', type=float, default=1.0, help='weight of adv obj')
    parser.add_argument('--lambda_obj_cls', type=float, default=1.0, help='weight of adv obj')
    parser.add_argument('--lambda_z_rec', type=float, default=1.0, help='weight of z rec')
    parser.add_argument('--lambda_c_rec', type=float, default=1.0, help='weight of c rec')
    parser.add_argument('--lambda_img_rec', type=float, default=1.0, help='weight of image rec')
    parser.add_argument('--lambda_kl', type=float, default=1.0, help='weight of kl')
    parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight of domain-invariant perceptual loss')
    parser.add_argument('--lambda_tv', type=float, default=1.0, help='weight of the tv loss')
    parser.add_argument('--lambda_identity', type=float, default=1.0, help='weight of the identity loss')

    # Log setting
    parser.add_argument('--resume_iter', type=str, default='l', help='l: from latest; s: from scratch; xxx: from iteration xxx')
    parser.add_argument('--log_step', type=int, default=100, help='How often do you want to log the training stats')
    parser.add_argument('--tensorboard_step', type=int, default=500)
    parser.add_argument('--image_step', type=int, default=500)
    parser.add_argument('--save_step', type=int, default=50000, help='How often do you want to save output images during training')
    parser.add_argument('--use_tensorboard', type=str2bool, default='true')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=3,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--update_html_freq', type=int, default=1000,
                        help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

    # define trainer wrap
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--trainer', type=str, default='T2V', help="T2V|?")
    parser.add_argument('--vgg_model_path', type=str, default='models', help="vgg model path")
    parser.add_argument('--display_image_size', type=int, default=1, help="show the result samples for images number")
    parser.add_argument('--display_result_size', type=int, default=1, help="show the result for images/obj number")
    parser.add_argument('--init', type=str, default='orthogonal', help="initialization [gaussian/kaiming/xavier/orthogonal")

    # training tricks:
    # 1, distributed traning setup: each process runs on 1 GPU device specified by the local_rank argument.
    parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--accumulation_steps", type=int, default=16, help="gradient accumulation for gpu out of memory")

    config = parser.parse_args()
    config.exp_name = 'results_{}'.format(config.dataset)

    # print(config)
    # print(config.local_rank)

    main(config)
