from options.test_options import TestOptions
from models import create_model
import os
from common_path import project_path_

'========================================================Attack both template and search================================================'
opt0 = TestOptions().parse()
opt1 = TestOptions().parse()
print(opt0,opt1)

opt0.name = 'G_template_L2_500'
opt0.model = 'G_template_L2_500'
opt0.netG = 'unet_128'
opt1.name = 'G_search_L2_500'
opt1.model = 'G_search_L2_500'
opt1.netG = 'unet_256'

# create and initialize model
'''create perturbation generator'''
GAN_0 = create_model(opt0)  # create a model given opt.model and other options
GAN_0.load_path = os.path.join(project_path_+'/checkpoints/'+'G_template_L2_500'+'/latest_net_G.pth')
GAN_0.setup(opt0)  # # regular setup: load and print networks; create schedulers
GAN_0.eval()



# create and initialize model
'''create perturbation generator'''
GAN_1 = create_model(opt1)  # create a model given opt.model and other options
GAN_1.load_path = os.path.join(project_path_+'/checkpoints/'+'G_search_L2_500'+'/model.pth')
GAN_1.setup(opt1)  # # regular setup: load and print networks; create schedulers
GAN_1.eval()
'========================================================Attack both template and search================================================'


