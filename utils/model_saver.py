import os
import re
import torch
from pathlib import Path

from torchvision import utils as vutils

import PIL
from PIL import Image


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    # input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)


def prepare_dir(DIR, name):
    gen_log_save_dir = os.path.join(DIR, 'checkpoints/{}/gen_logs'.format(name))
    dis_log_save_dir = os.path.join(DIR, 'checkpoints/{}/dis_logs'.format(name))
    model_save_dir = os.path.join(DIR, 'checkpoints/{}/models'.format(name))
    result_save_dir = os.path.join(DIR, 'checkpoints/{}/results'.format(name))

    if not Path(gen_log_save_dir).exists():
        Path(gen_log_save_dir).mkdir(parents=True)
    if not Path(dis_log_save_dir).exists():
        Path(dis_log_save_dir).mkdir(parents=True)
    if not Path(model_save_dir).exists():
        Path(model_save_dir).mkdir(parents=True)
    if not Path(result_save_dir).exists():
        Path(result_save_dir).mkdir(parents=True)

    return gen_log_save_dir, dis_log_save_dir, model_save_dir, result_save_dir


def load_model(model, model_dir=None, appendix=None, iter='l', map_location=None):

    load_iter = None
    load_model = None

    if iter == 's' or not os.path.isdir(model_dir) or len(os.listdir(model_dir)) == 0:
        load_iter = 0
        # if not os.path.isdir(model_dir):
        #     print('models dir not exist')
        # elif len(os.listdir(model_dir)) == 0:
        #     print('models dir is empty')

        # print('train from scratch.')
        return load_iter

    # load latest epoch
    if iter == 'l':
        for file in os.listdir(model_dir):
            if appendix is not None and appendix not in file:
                continue

            if file.endswith('.pkl'):
                current_iter = re.search('iter-\d+', file).group(0).split('-')[1]

                if len(current_iter) > 0:
                    current_iter = int(current_iter)

                    if load_iter is None or current_iter > load_iter:
                        load_iter = current_iter
                        load_model = os.path.join(model_dir, file)
                else:
                    continue

        if load_iter is not None:
            print('load from iter: %d' % load_iter)
        else:
            print('iter is None!')
        model.load_state_dict(torch.load(load_model, map_location=map_location))

        return load_iter
    # from given iter
    else:
        iter = int(iter)
        for file in os.listdir(model_dir):
            if file.endswith('.pkl'):
                current_iter = re.search('iter-\d+', file).group(0).split('-')[1]
                if len(current_iter) > 0:
                    if int(current_iter) == iter:
                        load_iter = iter
                        load_model = os.path.join(model_dir, file)
                        break
        if load_model:
            model.load_state_dict(torch.load(load_model))
            if load_iter is not None:
                print('load from iter: %d' % load_iter)
            else:
                print('iter is None!')
        else:
            load_iter = 0
            print('there is not saved models of iter %d' % iter)
            print('train from scratch.')
        return load_iter


def save_model(model, model_dir=None, appendix=None, iter=1, save_num=5, save_step=1000):
    iter_idx = range(iter, iter - save_num * save_step, -save_step)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for file in os.listdir(model_dir):
        if file.endswith('.pkl'):
            current_iter = re.search('iter-\d+', file).group(0).split('-')[1]
            if len(current_iter) > 0:
                if int(current_iter) not in iter_idx:
                    pass
                    # print("should remove file: {} to keep only {} files exit".format(os.path.join(model_dir, file),
                    #                                                                  save_num))
                    os.remove(os.path.join(model_dir, file))
            else:
                continue

    if appendix:
        model_name = os.path.join(model_dir, 'iter-%d_%s.pkl' % (iter, appendix))
    else:
        model_name = os.path.join(model_dir, 'iter-%d.pkl' % iter)
    torch.save(model.state_dict(), model_name)


def save_model_bak(image, model_dir=None, appendix=None, iter=1, save_num=5, save_step=1000):
    iter_idx = range(iter, iter - save_num * save_step, -save_step)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for file in os.listdir(model_dir):
        if file.endswith('.jpg'):
            current_iter = re.search('iter-\d+', file).group(0).split('-')[1]
            if len(current_iter) > 0:
                if int(current_iter) not in iter_idx:
                    print("should remove file: {} to keep only {} files exit".format(os.path.join(model_dir, file), save_num))
                    os.remove(os.path.join(model_dir, file))
            else:
                continue

    if appendix:
        model_name = os.path.join(model_dir, 'iter-%d_%s.jpg' % (iter, appendix))
    else:
        model_name = os.path.join(model_dir, 'iter-%d.jpg' % iter)
    image.save(model_name)
