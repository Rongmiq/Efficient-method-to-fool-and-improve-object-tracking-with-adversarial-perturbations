import os

import torch
from .base_model import BaseModel
from . import networks
import math
import numpy as np
import argparse

from siamRPNPP import SiamRPNPP
from data_utils import normalize
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F
from PIL import Image


parser = argparse.ArgumentParser(description='siamcar_r50 tracking')
parser.add_argument('--tracker_name', type=str, default='siamcar_otb',help='tracker name such as siamcar, siamgat ....')
args = parser.parse_args()

class GtemplateL2500Model(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=500, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G', 'G_L2','cos0', 'cos1', 'cos2', 'cos3', 'cos4', 'cos5']
        self.loss_names = ['G', 'G_L2', 'G_Linf','cos', 'dis']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['template_clean1', 'template_adv1']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            self.init_weight = 0.1
            self.margin = -5
            self.margin_min = 0.1
            self.margin_max = 1.0
            self.weight = self.init_weight
            self.init_weight_L2 = opt.lambda_L1
            self.weight_L2 = self.init_weight_L2

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        '''siamrpn++'''
        self.siam = SiamRPNPP(tracker=args.tracker_name)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.template_clean255 = input[0].squeeze(0).cuda()  # pytorch tensor, shape=[1,3,127,127] [0,255]
        self.template_clean1 = normalize(self.template_clean255)  # shape=[1,3,127,127]  (0,255) ---> (-1,1)
        self.search_clean255 = input[1].squeeze(0).cuda()  # pytorch tensor, shape=[N,3,255,255]

    def tensorshow(self, tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''pad to (128,128)'''
        template128_clean = torch.zeros(1, 3, 128, 128).cuda()
        template128_clean[:, :, 1:, 1:] = self.template_clean1
        template128_adv = template128_clean +  self.netG(template128_clean)  # Residual form: G(A)+A

        '''Then crop back to (127,127)'''
        self.template_adv1 = template128_adv[:, :, 1:, 1:]
        self.template_adv255 = self.template_adv1 * 127.5 + 127.5  # [0,255]


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.template_adv1, self.template_clean1) * self.weight_L2
        self.loss_G_Linf = self.linf(self.template_adv255,self.template_clean255)

        n,c,w,h = self.clean_responses.shape
        clean_response = self.clean_responses.view(n, c, w * h)
        adv_response = self.adv_responses.view(n, c, w * h)
        dis = torch.sum(torch.abs(self.adv_cen_x-self.clean_cen_x)) + torch.sum(torch.abs(self.adv_cen_y-self.clean_cen_y))

        cos = 1 - torch.cosine_similarity(-clean_response/2, adv_response, dim=-2)  # [n,256]
        cos_loss = torch.mean(torch.mean(cos ,dim=-1))

        self.loss_cos = cos_loss
        self.loss_dis = (25 - dis) * 0.1

        self.loss_G = self.loss_G_L2 + self.loss_cos * 2 + self.loss_dis + self.loss_G_Linf
        self.loss_G.backward()

    def optimize_parameters(self):
        '''One forward & backward pass. One update of parameters'''
        '''forward pass'''
        # 1. predict with clean template
        with torch.no_grad():
            self.siam.model.template(self.template_clean255)
            self.clean_cen_x, self.clean_cen_y,self.clean_responses = self.siam.get_cen_loc(self.search_clean255)
        # 2. adversarial attack with GAN
        self.forward()  # compute fake image
        # 3. predict with adversarial template
        self.siam.model.template(self.template_adv255)
        self.adv_cen_x, self.adv_cen_y,self.adv_responses = self.siam.get_cen_loc(self.search_clean255)
        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def linf(self, adv_img, clean_img):
        l = torch.zeros(1)
        G_img = adv_img.unsqueeze(0) - clean_img.unsqueeze(0)
        G_img = G_img.view(G_img.shape[0], -1)
        G_img_np = G_img.detach().clone().cpu().numpy()

        L_inf = np.linalg.norm(G_img_np, ord=np.inf)
        L_inf = L_inf / G_img.shape[1]
        L_inf = np.abs(L_inf)
        l = L_inf
        return l





