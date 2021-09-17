from options.test_options import TestOptions
from models import create_model
import os
from common_path import project_path_
opt = TestOptions().parse()

# modify some config
'''Attack Template'''
opt.model = 'G_template_L2_500' # only cooling
# opt.model = 'G_template_L2_500_regress' # cooling + shrinking
# opt.model = 'G_template_L2_1000'
opt.netG = 'unet_128'


# create and initialize model
'''create perturbation generator'''
GAN = create_model(opt)  # create a model given opt.model and other options
# GAN.load_path = os.path.join(project_path_+'/checkpoints/'+opt.model+'/latest_net_G.pth')
GAN.load_path = os.path.join(project_path_+'/checkpoints/'+opt.model+'/model.pth')
GAN.setup(opt)  # # regular setup: load and print networks; create schedulers
GAN.eval()