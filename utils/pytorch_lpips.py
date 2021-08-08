import argparse
import lpips


class util_of_lpips():
    def __init__(self, net, use_gpu=False):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
        '''

        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py
        '''

        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01


def main(opt):
    ## Initializing the model
    loss_fn = lpips.LPIPS(net='alex', version=opt.version)

    if(opt.use_gpu):
        loss_fn.cuda()

    # Load images
    img0 = lpips.im2tensor(lpips.load_image(opt.path0))  # RGB image from [-1,1]
    img1 = lpips.im2tensor(lpips.load_image(opt.path1))

    if(opt.use_gpu):
        img0 = img0.cuda()
        img1 = img1.cuda()

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)
    print('Distance: %.3f' % dist01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p0', '--path0', type=str, default='../../TICGAN/epochs/TICGAN_KAIST/test/C_0.jpg')
    parser.add_argument('-p1', '--path1', type=str, default='../../datasets/KAIST-MS/thermal2visible_256x256/testV/0.jpg')
    parser.add_argument('-v', '--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

    opt = parser.parse_args()
    main(opt)
