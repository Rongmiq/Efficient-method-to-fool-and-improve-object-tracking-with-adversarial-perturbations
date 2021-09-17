# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from torchvision import transforms
from scipy import stats
import os
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain
from common_path import project_path_
from torchvision import utils as vutils
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

'''Capsule SiamRPN++(We can use it as one component in higher-level task)'''


class SiamRPNPP():
    def __init__(self, tracker=''):
        if 'siamrpn++_otb' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth')
        elif 'siamrpn++_vot' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_r50_l234_dwxcorr/model.pth')
        elif 'siamcar_otb' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/model.pth')
        elif 'siamcar_lasot' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/LaSOT_model.pth')
        elif 'siamrpn_otb' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_alex_dwxcorr_otb/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_alex_dwxcorr_otb/model.pth')
        elif 'siamrpn' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_alex_dwxcorr/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_alex_dwxcorr/model.pth')
        elif 'siammask' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siammask_r50_l3/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siammask_r50_l3/model.pth')
        elif 'siamgat_otb' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamgat_googlenet/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamgat_googlenet/model.pth')
        elif 'siamgat_lasot' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamgat_googlenet_lasot/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamgat_googlenet_lasot/model.pth')
        elif 'siamban_otb' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamban_r50/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamban_r50/model.pth')
        elif 'siamban_vot' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamban_r50/config_vot.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamban_r50/model_vot.pth')
        elif 'siamrpn_alex_dwxcorr_otb' in tracker:
            cfg_file = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_alex_dwxcorr_otb/config.yaml')
            snapshot = os.path.join(project_path_, '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_alex_dwxcorr_otb/model.pth')
        else:
            raise ValueError('tracker name unknown!')
        print('===========================================================================')
        print('load model from :', snapshot)
        print('===========================================================================')

        # load config
        cfg.merge_from_file(cfg_file)
        # create model
        self.model = ModelBuilder(tracker)  # A Neural Network.(a torch.nn.Module)
        # load model
        self.model = load_pretrain(self.model, snapshot).cuda().eval()

    def get_heat_map(self, X_crop, softmax=False):
        score_map = self.model.track(X_crop)['cls']  # (N,2x5,25,25)
        score_map = score_map.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # (5HWN,2)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]  # (5HWN,)
        return score_map

    def get_cls_reg(self, X_crop, softmax=False):
        outputs = self.model.track(X_crop)  # (N,2x5,25,25)
        score_map = outputs['cls'].permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)  # (5HWN,2)
        reg_res = outputs['loc'].permute(1, 2, 3, 0).contiguous().view(4, -1)
        if softmax:
            score_map = F.softmax(score_map, dim=1).data[:, 1]  # (5HWN,)
        return score_map, reg_res

    def fast_xcorr(self, z_img, x_img):
        # fast cross correlation
        z = self.get_z_features(z_img)
        x = self.get_x_features(x_img)
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        out_np = out.cpu().detach().numpy()
        out_max = out_np.max()
        out = out_np / (out_max + 0.000001)
        responses = torch.from_numpy(out).cuda()
        responses = responses.squeeze(1)  # [n,25,25]
        responses = responses.view(responses.shape[0], -1)  # [n,625]

        return responses

    def get_responses(self, z, x):
        responses = self.fast_xcorr(z, x)  # [n,1,25,25]

        return responses

    def get_cen_loc(self, X_crop):
        output_dict = self.model.track(X_crop) # [N,1,25,25]
        cen_maps = output_dict['cen']
        features = output_dict['features']

        N = cen_maps.shape[0]

        loc_x = torch.zeros(N)
        loc_y = torch.zeros(N)
        # sec_xy = []
        for i in range(0,N):

            cen_map = cen_maps[i] #[1,25,25]
            cen_map = torch.squeeze(cen_map) #[1,1,25,25] -> [25,25]
            cen_map_np = cen_map.detach().cpu().numpy()
            ind_max = np.argmax(cen_map_np)
            pre_loc = np.unravel_index(ind_max, cen_map_np.shape)
            loc_x[i] = float(pre_loc[0])
            loc_y[i] = float(pre_loc[1])


        return loc_x,loc_y,features

    def cen_loc_index_value(self, X_crop):
        # print('crop_shape',X_crop.shape)
        cen_maps, cls_maps = self.get_out_maps(X_crop)
        N = cen_maps.shape[0]
        loc_xy = []
        loc_xy_sec = []
        value1 = torch.zeros(N)
        value2 = torch.zeros(N)
        for i in range(0, N):
            # cen_map = torch.squeeze(cen_map[i]) #[1,1,25,25] -> [25,25]

            cen_map = cen_maps[i]
            # print(cen_map.shape)
            cen = cen_map[0]
            cen_copy = cen.clone()
            # print(cen)
            cen_numpy = cen.data.cpu().numpy()
            loc = np.unravel_index(cen_numpy.argmax(), cen_numpy.shape)

            loc_xy.append(loc)
            value1[i] = cen[loc[0]][loc[1]]

            cen_copy[loc[0]][loc[1]] = 0
            cen_numpy_copy = cen_copy.data.cpu().numpy()
            sec_max = np.unravel_index(cen_numpy_copy.argmax(), cen_numpy_copy.shape)
            loc_xy_sec.append(sec_max)

            value2[i] = cen[sec_max[0]][sec_max[1]]
            # loc_c.append(max_c_up)_

        if len(loc_xy) != cls_maps.shape[0]:
            raise ValueError('error! lec(loc_r) != N')

        return loc_xy, value1, loc_xy_sec, value2


    def dis_from_indexes(self, xcorr_clean_features, xcorr_adv_features):

        score = 1
        clean_features = (xcorr_clean_features * score).view(xcorr_clean_features.shape[0],
                                                             xcorr_clean_features.shape[1], -1)
        adv_features = (xcorr_adv_features * score).view(xcorr_clean_features.shape[0], xcorr_clean_features.shape[1],
                                                         -1)
        cos = 1 + torch.cosine_similarity(clean_features, adv_features, dim=-1)  # [n,256]
        cos148 = cos[:, 148]
        cos222 = cos[:, 222]
        cos226 = cos[:, 226]
        cos148_acg = torch.mean(cos148)
        cos222_acg = torch.mean(cos222)
        cos226_acg = torch.mean(cos226)

        return cos148_acg, cos222_acg, cos226_acg

    def get_x_features(self, X_crop):

        x_featutres = self.model.search(X_crop)  # list len=3
        x_featutre0 = x_featutres[0]
        x_featutre1 = x_featutres[1]
        x_featutre2 = x_featutres[2]
        x_featutres_avg = (x_featutre0 + x_featutre1 + x_featutre2) / 3

        return x_featutres_avg

    def get_z_features(self, z):

        z_featutres = self.model.template(z)  # list len=3
        z_featutre0 = z_featutres[0]
        z_featutre1 = z_featutres[1]
        z_featutre2 = z_featutres[2]
        z_featutres_avg = (z_featutre0 + z_featutre1 + z_featutre2) / 3
        return z_featutres_avg

    def get_x_features_list(self, X_crop):

        x_featutres = self.model.search(X_crop)  # list len=3

        return x_featutres

    def get_z_features_list(self, z):

        z_featutres = self.model.template(z)  # list len=3

        return z_featutres


    def similarity(self, clean_xcorr_features, adv_xcorr_features):
        '''

        Args:
            clean_features:
            adv_features:

        Returns:
            Cosine Similarity in dim1(channels)
        '''
        num = clean_xcorr_features.shape[0]  # N

        similarity_sum = []
        for i in range(0, num):  # [n,625]
            similarity = (1 + torch.cosine_similarity(clean_xcorr_features[i], adv_xcorr_features[i], dim=0))  # [625]
            similarity_avg = torch.mean(similarity)
            similarity_sum.append(similarity_avg)
        similarity_sum_avg = sum(similarity_sum) / len(similarity_sum)
        return similarity_sum_avg

    def dis(self, clean_features, adv_features):
        '''

        Args:
            features: adv features or clean feature after view to [cwh,1]
                type: list,  len = 3

        Returns:
            dis between adv features and clean features
        '''

        num0 = clean_features.shape[0]
        dis = []
        for i in range(num0):
            dis_c = F.pairwise_distance(torch.transpose(clean_features[i], 0, 1),
                                        torch.transpose(adv_features[i], 0, 1), p=2)  # [625]
            dis_c_avg = torch.mean(dis_c)
            dis.append(dis_c_avg)
        dis_avg = sum(dis) / num0
        return dis_avg

    def similarity_c(self, clean_features, adv_features):
        '''

        Args:
            clean_features:
            adv_features:

        Returns:
            Cosine Similarity in dim1(channels)
        '''
        num = clean_features.shape[0]
        num1 = clean_features.shape[1]

        clean_feature = clean_features.contiguous().view(-1)  # [n,256,wh]
        adv_feature = adv_features.contiguous().view(-1)  # [n,256,wh]
        similarity_sum = []
        for i in range(0, num):  # [256,wh]
            similarity = (1 + torch.cosine_similarity(clean_feature[i], adv_feature[i], dim=0))  # [625]
            similarity_avg = torch.mean(similarity)
            similarity_sum.append(similarity_avg)
        similarity_sum_avg = sum(similarity_sum) / len(similarity_sum)
        return similarity_sum_avg

    def similarity_wh(self, clean_features, adv_features):
        '''

        Args:
            clean_features:
            adv_features:

        Returns:
            Cosine Similarity in dim1(channels)
        '''
        num = clean_features.shape[0]
        num1 = clean_features.shape[1]

        clean_feature = clean_features.contiguous().view(num, num1, -1)  # [n,256,wh]
        adv_feature = adv_features.contiguous().view(num, num1, -1)  # [n,256,wh]
        similarity_sum = []
        for i in range(0, num):  # [256,wh]
            similarity = (1 + torch.cosine_similarity(clean_feature[i], adv_feature[i], dim=1))  # [256]
            similarity_avg = torch.mean(similarity)
            similarity_sum.append(similarity_avg)
        similarity_sum_avg = sum(similarity_sum) / num
        return similarity_sum_avg

    def similarity_three_layers(self, clean_features_list, adv_features_list):
        similarity1 = self.similarity_c(clean_features_list[0], adv_features_list[0])
        similarity2 = self.similarity_c(clean_features_list[1], adv_features_list[1])
        similarity3 = self.similarity_c(clean_features_list[2], adv_features_list[2])

        return similarity1, similarity2, similarity3  # len = 1

    def Lp_max(self, features):
        features_3 = features.view(features.shape[0], features.shape[1], -1)
        adv_features = features_3.data.cpu().numpy()  # [n,256,wh]
        Lp_max = []
        Lp_avg = []
        for i in range(0, adv_features.shape[0]):  # Nx[256,wh]
            for j in range(0, adv_features.shape[1]):  # 256x[wh,1]
                l = np.linalg.norm(adv_features[i][j], ord=np.inf)
                avg = np.mean(adv_features[i])
                Lp_max.append(l)
                Lp_avg.append(avg)
            print(len(Lp_max))
            print(len(Lp_avg))
        Lp_max_avg = sum(Lp_max) / int(len(Lp_max))
        Lp_avg_avg = sum(Lp_avg) / int(len(Lp_avg))
        return Lp_max_avg, Lp_avg_avg

