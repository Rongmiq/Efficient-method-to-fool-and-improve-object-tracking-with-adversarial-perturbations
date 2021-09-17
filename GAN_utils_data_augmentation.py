from options.test_options import TestOptions
from models import create_model
import os
from common_path import project_path_


'==================================================''Attack'' both template and search for data augmentation================================================'
opt0 = TestOptions().parse()
print(opt0)

opt0.name = 'G_search_L2_1000_cen'
opt0.model = 'G_search_L2_1000_cen'
opt0.netG = 'unet_256'

# create and initialize model
'''create perturbation generator'''
GAN_0 = create_model(opt0)  # create a model given opt.model and other options
GAN_0.load_path = os.path.join(project_path_+'/checkpoints/'+'G_search_L2_1000_cen_model'+'/model.pth')
GAN_0.setup(opt0)  # # regular setup: load and print networks; create schedulers
GAN_0.eval()

'==================================================''Attack'' both template and search for data augmentation================================================'
