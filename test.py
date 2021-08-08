import torch
import argparse
from models.generator import Generator

import math
from tqdm import tqdm

from data.t2v_custom_mask import get_dataloader as get_dataloader_t2v
from utils.data import validation_methods, calculation_lpips, convert_normalize, convert_transform, image_show_normal
from utils.panoptic_proccessing import get_panoptic_data_img

import os

from pathlib import Path
import torch.backends.cudnn as cudnn

from utils.miscs import str2bool

from utils.data import image_save

import PIL
from PIL import Image
import matplotlib.pyplot as plt

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

    result_save_dir = os.path.join(config.dir, config.result_dir)
    if not Path(result_save_dir).exists():
        Path(result_save_dir).mkdir(parents=True)

    test_loader = get_dataloader_t2v(batch_size=config.batch_size, T2V_DIR=config.t2v_dir,
                                     is_training=config.is_training)

    test_bar = tqdm(test_loader, desc='Thermal2Visible')
    # running_results = {'batch_sizes': 0, 'psnr': 0, 'rmse': 0, 'ssim': 0, 'lpips': 0}

    # category_num = 134, from panoptic 2017 annotation, stuffs: 80 ~ 133, things: 0 ~ 79, so total: 134:
    vocab_num = config.category_num

    assert config.clstm_layers > 0

    # 生成网络模型t, # auto-encoder for domain thermal
    netG_t = Generator(class_num=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim,
                       c_dim=config.c_dim, clstm_layers=config.clstm_layers).cuda()
    # 生成网络模型v, # auto-encoder for domain visible
    netG_v = Generator(class_num=vocab_num, embedding_dim=config.embedding_dim, z_dim=config.z_dim,
                       c_dim=config.c_dim, clstm_layers=config.clstm_layers).cuda()

    # distributed mode wrapping
    netG_t = DistributedDataParallel(netG_t, device_ids=[local_rank], output_device=local_rank)
    netG_v = DistributedDataParallel(netG_v, device_ids=[local_rank], output_device=local_rank)

    map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    netG_t.load_state_dict(torch.load(config.saved_model_t, map_location=map_location), False)
    netG_v.load_state_dict(torch.load(config.saved_model_v, map_location=map_location), False)

    # netG_t.load_state_dict(torch.load(config.saved_model_t))
    # netG_v.load_state_dict(torch.load(config.saved_model_v))

    # data_iter = iter(test_loader)

    with torch.no_grad():
        netG_t.eval()
        netG_v.eval()

        # print("GPU: {}, Testing Start ...".format(local_rank))
        #
        # for i in range(len(test_loader.dataset)):
        #     try:
        #         batch = next(data_iter)
        #     except:
        #         data_iter = iter(test_loader)
        #         batch = next(data_iter)

        for batch in enumerate(test_bar):
            # 指定数据存储计算的设备
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            img_t, img_v, objs, boxes, masks, obj_to_img, fname = batch[1]
            style_rand_t = torch.randn(objs.size(0), config.z_dim)  # Random Norm Distribution and style dim: 256
            style_rand_v = torch.randn(objs.size(0), config.z_dim)  # Random Norm Distribution and style dim: 256
            img_t, img_v, objs, boxes, masks, obj_to_img, style_rand_t, style_rand_v = img_t.cuda(), img_v.cuda(), \
                   objs.cuda(), boxes.cuda(), masks.cuda(), obj_to_img.cuda(), style_rand_t.cuda(), style_rand_v.cuda()

            # =================================================================================== #
            #                             2. inference networks                                   #
            # =================================================================================== #
            # Generate fake image
            # 1, Object Style and Content Encoder:
            content_t, style_t, mu_t, logvar_t = netG_t.module.encode(img_t, objs, boxes)

            # 2, decode (cross domain) 进行交叉解码，即两张图片的content code，style code进行互换
            img_t2v = netG_v.module.decode(content_t, style_rand_v, objs, boxes, masks, obj_to_img)

            print(result_save_dir + fname)

            image_save('%s' % fname, img_t2v, result_save_dir)

            # running_results['batch_sizes'] += config.batch_size

            # image_show_normal(img_t2v)
            # image_show_normal(img_v)

            # # [-1, 1] to [0, 1]
            img_t2v = convert_transform(img_t2v).unsqueeze(0).cuda()
            # img_v = convert_transform(img_v).unsqueeze(0).cuda()

            # mse, batch_ssim, psnr = validation_methods(img_t2v, img_v)
            # running_results['ssim'] += batch_ssim
            # running_results['psnr'] += psnr
            # running_results['rmse'] += math.sqrt(mse)

            # running_results['ssim'] += 0
            # running_results['psnr'] += 0
            # running_results['rmse'] += 0

            # ndarr = img_t2v[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            # im = Image.fromarray(ndarr)
            # plt.imshow(im)
            # plt.show()

            # img_t2v_path = os.path.join(images_dir, 'C_' + fname)
            # img_t2v_path = os.path.join(images_dir, '%d.jpg' % (id))
            # im.save(img_t2v_path)
            # img_v_path = os.path.join(config.t2v_dir + '/testV', fname)
            # img_v_path = os.path.join(config.t2v_dir + '/testV', '%d.jpg' % (id))

            # lpips = calculation_lpips(img_t2v_path, img_v_path)
            # running_results['lpips'] += lpips
            #
            # test_bar.set_description(desc='PSNR: %.6f  SSIM: %.6f   RMSE: %.6f   LPIPS: %.6f'
            #                               % (running_results['psnr'] / running_results['batch_sizes'],
            #                                  running_results['ssim'] / running_results['batch_sizes'],
            #                                  running_results['rmse'] / running_results['batch_sizes'],
            #                                  running_results['lpips'] / running_results['batch_sizes']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Datasets configuration
    parser.add_argument('--dataset', type=str, default='thermal2visible_256x256', help='dataset selection')
    parser.add_argument('--dir', type=str, default='/tmp/OL-GANs')
    parser.add_argument('--t2v_dir', type=str, default='/tmp/OL-GANs/datasets/thermal2visible')
    parser.add_argument('--is_training', type=str2bool, default='false')
    parser.add_argument('--batch_size', type=int, default=1, help='select batch size')

    # Model configuration
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--category_num', type=int, default=133, help='stuffs: 80 ~ 133, things: 0 ~ 79, so total: 134')
    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--c_dim', type=int, default=512)
    parser.add_argument('--clstm_layers', type=int, default=3)

    # distributed testing setup: each process runs on 1 GPU device specified by the local_rank argument.
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    # Model setting
    parser.add_argument('--saved_model_t', type=str,
                        default='/tmp/OL-GANs/checkpoints/pretrained/models/iter-400000_netG_t.pkl')
    parser.add_argument('--saved_model_v', type=str,
                        default='/tmp/OL-GANs/checkpoints/pretrained/models/iter-400000_netG_v.pkl')
    parser.add_argument('--result_dir', type=str, default='tests')

    config = parser.parse_args()

    # print(config)

    main(config)
