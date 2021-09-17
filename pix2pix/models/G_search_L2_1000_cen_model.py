import torch
from .base_model import BaseModel
from . import networks

'''siamrpn++'''
from siamRPNPP import SiamRPNPP
from data_utils import normalize


class GsearchL21000CenModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L2', type=float, default=1500, help='weight for L2 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_L2', 'cen', 'cen_1','cen_2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.visual_names = ['search_clean_vis','search_adv_vis']
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
            self.weight = self.init_weight
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        '''siamrpn++'''
        self.siam = SiamRPNPP(tracker='siamcar_otb')

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.template_clean127 = input[0].squeeze(0).cuda()  # pytorch tensor, shape=(1,3,127,127)
        self.template_clean127_1 = normalize(self.template_clean127)
        self.search_clean255 = input[1].squeeze(0).cuda() # pytorch tensor, shape=(N,3,255,255) [0,255]
        self.search_clean255_1 = normalize(self.search_clean255)
        self.num_search = self.template_clean127.size(0)
        self.train = True


    def forward(self,target_sz=(255,255)):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.train:
            self.template_clean256_1 = torch.nn.functional.interpolate(self.template_clean127_1, size=(256,256), mode='bilinear')
            self.template_adv256_1 = self.template_clean256_1 + self.netG(self.template_clean256_1)
            self.template_adv127_1 = torch.nn.functional.interpolate(self.template_adv256_1, size=(127, 127),mode='bilinear')
            self.template_adv127_255 = self.template_adv127_1 * 127.5 + 127.5

            self.search_clean256_1 = torch.nn.functional.interpolate(self.search_clean255_1, size=(256,256), mode='bilinear')
            self.search_adv256_1 = self.search_clean256_1 + self.netG(self.search_clean256_1)
            self.search_adv255_1 = torch.nn.functional.interpolate(self.search_adv256_1, size=(255, 255),mode='bilinear')
            self.search_adv255_255 = self.search_adv255_1 * 127.5 + 127.5

            self.search_clean_vis = self.search_clean255_1[0:1]
            self.search_adv_vis = self.search_adv255_1[0:1]

        else:
            if self.clean_img.shape[-1] == 127:
                self.template_clean127_1 = self.clean_img
                template_clean512_1 = torch.nn.functional.interpolate(self.template_clean127_1, size=(256,256), mode='bilinear')
                template_adv512_1 = template_clean512_1 + self.netG(template_clean512_1)  # Residual form: G(A)+A
                '''Then crop back to (127,127)'''
                self.template_adv127_1 = torch.nn.functional.interpolate(template_adv512_1, size=(127,127), mode='bilinear')
                self.template_adv127_255 = self.template_adv127_1 * 127.5 + 127.5
                self.img_adv = self.template_adv127_255

            elif self.clean_img.shape[-1] == 255:
                self.search_clean255_1 = self.clean_img
                search_clean512_1 = torch.nn.functional.interpolate(self.search_clean255_1, size=(256,256), mode='bilinear')
                search_adv512_1 = search_clean512_1 + self.netG(search_clean512_1)  # Residual form: G(A)+A
                '''Then crop back to (255,255)'''
                self.search_adv255_1 = torch.nn.functional.interpolate(search_adv512_1, size=target_sz, mode='bilinear')
                self.search_adv255_255 = self.search_adv255_1 * 127.5 + 127.5
                self.img_adv = self.search_adv255_255


    def backward_G(self):
        """Calculate GAN and L2 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L2 = (self.criterionL2(self.template_clean127_1, self.template_adv127_1) + self.criterionL2(self.search_clean256_1, self.search_adv256_1))* self.opt.lambda_L2

        self.loss_cen_1 = torch.mean(self.adv_value1)
        self.loss_cen_2 = torch.mean(self.clean_value1)
        self.loss_cen_2 = torch.mean(self.adv_value2 - self.clean_value2).detach().cpu().numpy()

        self.loss_cen = max((self.loss_cen_2 - self.loss_cen_1) * 100,-3)
        self.loss_G = self.loss_G_L2 + self.loss_cen
        self.loss_G.backward()

    def optimize_parameters(self):
        '''One forward & backward pass. One update of parameters'''
        '''forward pass'''
        # 1. predict with clean template
        with torch.no_grad():
            self.siam.model.template(self.template_clean127)
            _, self.clean_value1, _, self.clean_value2 = self.siam.cen_loc_index_value(self.search_clean255)
        # 2. adversarial attack with GAN
        self.forward()  # compute fake image
        # 3. predict with adversarial template
        self.siam.model.template(self.template_adv127_255)
        _, self.adv_value1, _, self.adv_value2 = self.siam.cen_loc_index_value(self.search_adv255_255)
        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def linf(self, adv_img, clean_img):

        G_img = adv_img - clean_img
        G_img = G_img.view(G_img.shape[0], -1)
        G_img_np = G_img.detach().clone().cpu().numpy()

        L_inf = np.linalg.norm(G_img_np, ord=np.inf)
        L_inf = L_inf / G_img.shape[0] / G_img.shape[1]

        return L_inf
