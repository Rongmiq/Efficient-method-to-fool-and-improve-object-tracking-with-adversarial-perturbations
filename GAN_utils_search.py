from options.test_options import TestOptions
from models import create_model
import os
from common_path import project_path_
opt = TestOptions().parse()
print(opt)
# # modify some config
# '''Attack Template patch'''
# # opt.model = 'G_search_L2_1000_cen' # only cooling
# # opt.model = 'G_search_L2_500' # only cooling
# opt.model = 'G_template_L2_500' # only cooling
# opt.name = 'G_template_L2_500' # only cooling
# # opt.model = 'G_template_search_L2_500'
# # opt.model = 'G_search_L2_500'
# opt.netG= 'unet_128'
#
# # create and initialize model
# '''create perturbation generator'''
# GAN_0 = create_model(opt)  # create a model given opt.model and other options
# GAN_0.load_path = os.path.join(project_path_+'/checkpoints/'+opt.model+'/latest_net_G.pth')
# GAN_0.setup(opt)  # # regular setup: load and print networks; create schedulers
# GAN_0.eval()



# modify some config
'''Attack Search Regions'''
# opt.model = 'G_search_L2_1000_cen' # only cooling
# opt.model = 'G_search_L2_500' # only cooling
# opt.model = 'G_template_L2_500' # only cooling
# opt.model = 'G_template_search_L2_500'
opt.name = 'G_search_L2_500'
opt.model = 'G_search_L2_500'
opt.netG = 'unet_256'

# create and initialize model
'''create perturbation generator'''
GAN_1 = create_model(opt)  # create a model given opt.model and other options
GAN_1.load_path = os.path.join(project_path_+'/checkpoints/'+opt.model+'/model.pth')
GAN_1.setup(opt)  # # regular setup: load and print networks; create schedulers
GAN_1.eval()
