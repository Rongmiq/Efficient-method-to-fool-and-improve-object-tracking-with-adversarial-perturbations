# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import torch.nn.functional as F
import torch
import cv2

from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip
from attack_utils import adv_attack_template, adv_attack_search, adv_attack_search_gat, \
    add_gauss_noise, add_pulse_noise, adv_attack_template_S
from data_utils import tensor2img


class SiamGATTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamGATTracker, self).__init__()
        self.score_size = cfg.TRACK.SCORE_SIZE
        hanning = np.hanning(self.score_size)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()

    def _convert_cls(self, score):
        score = F.softmax(score[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        score += 0.05
        return score

    def _convert_cen(self, cen):
        cen = cen.data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        return cen

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    # replace by get_z_crop()
    '''
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        scale = cfg.TRACK.EXEMPLAR_SIZE / s_z
        c = (cfg.TRACK.EXEMPLAR_SIZE - 1) / 2
        roi = torch.tensor([[c - bbox[2] * scale / 2, c - bbox[3] * scale / 2,
                             c + bbox[2] * scale / 2, c + bbox[3] * scale / 2]])
        self.model.template_gat(z_crop, roi)

    '''

    def init(self,img,bbox):
        z_crop,roi = self.get_z_crop(img,bbox)
        self.model.template_gat(z_crop, roi)

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, bboxes, penalty_lk, scale_z):
        bboxes_w = bboxes[0, :, :] + bboxes[2, :, :]
        bboxes_h = bboxes[1, :, :] + bboxes[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = cfg.TRACK.OFFSET
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, coarseSco, fineSco, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        CRowUp, CColUp = np.unravel_index(coarseSco.argmax(), coarseSco.shape)
        CRow = int(round(CRowUp / scale_score))
        CCol = int(round(CColUp / scale_score))
        CRow = bbox_clip(CRow, 0, cfg.TRACK.SCORE_SIZE-1)
        CCol = bbox_clip(CCol, 0, cfg.TRACK.SCORE_SIZE-1)
        bbox_region = lrtbs[CRow, CCol, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)

        l = int(min(CColUp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2)
        t = int(min(CRowUp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2)
        r = int(min(upsize - CColUp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2)
        b = int(min(upsize - CRowUp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2)

        mask = np.zeros_like(fineSco)
        mask[CRowUp - t: CRowUp + b + 1, CColUp - l: CColUp + r + 1] = 1
        fineSco = fineSco * mask
        return fineSco

    def getCenter(self, coarseSco, fineSco, scale_score, lrtbs, scale_z):
        # coarse location
        fineSco = self.coarse_location(coarseSco, fineSco, scale_score, lrtbs)
        # fine location
        CRowUp, CColUp = np.unravel_index(fineSco.argmax(), fineSco.shape)
        disp = self.accurate_location(CRowUp, CColUp)
        disp_ori = disp / scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return CRowUp, CColUp, new_cx, new_cy
    '''
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        ---------get_x_crop--------
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)

        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = self._convert_cen(outputs['cen']).squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, cfg.TRACK.PENALTY_K)
        p_cls = penalty * cls
        p_score = p_cls * cen

        if cfg.TRACK.hanming:
            hp_score = p_score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        lrtbs = np.transpose(lrtbs, (1, 2, 0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / (cfg.TRACK.SCORE_SIZE - 1)

        # get center
        CRowUp, CColUp, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)

        # get w h
        ave_w = (lrtbs_up[CRowUp, CColUp, 0] + lrtbs_up[CRowUp, CColUp, 2]) / self.scale_z
        ave_h = (lrtbs_up[CRowUp, CColUp, 1] + lrtbs_up[CRowUp, CColUp, 3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        lr = penalty * cls_up[CRowUp, CColUp] * cfg.TRACK.LR
        new_width = lr * ave_w + (1 - lr) * self.size[0]
        new_height = lr * ave_h + (1 - lr) * self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 0, img.shape[1])
        height = bbox_clip(new_height, 0, img.shape[0])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
        }
'''

    # replcae in track()
    def get_x_crop(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        return x_crop, scale_z

    def get_z_crop(self,img,bbox):

        """
            args:
                img(np.ndarray): BGR image
                bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                        bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average)
        scale = cfg.TRACK.EXEMPLAR_SIZE / s_z
        c = (cfg.TRACK.EXEMPLAR_SIZE - 1) / 2
        roi = torch.tensor([[c - bbox[2] * scale / 2, c - bbox[3] * scale / 2,
                                 c + bbox[2] * scale / 2, c + bbox[3] * scale / 2]])
        self.model.template_gat(z_crop, roi)

        return z_crop,roi

    def x_crop_2_res(self, img, x_crop, scale_z):

        outputs = self.model.track(x_crop)

        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = self._convert_cen(outputs['cen']).squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, cfg.TRACK.PENALTY_K,scale_z)
        p_cls = penalty * cls
        p_score = p_cls * cen

        if cfg.TRACK.hanming:
            hp_score = p_score * (1 - cfg.TRACK.WINDOW_INFLUENCE) + self.window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        lrtbs = np.transpose(lrtbs, (1, 2, 0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / (cfg.TRACK.SCORE_SIZE - 1)

        # get center
        CRowUp, CColUp, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs,scale_z)

        # get w h
        ave_w = (lrtbs_up[CRowUp, CColUp, 0] + lrtbs_up[CRowUp, CColUp, 2]) / scale_z
        ave_h = (lrtbs_up[CRowUp, CColUp, 1] + lrtbs_up[CRowUp, CColUp, 3]) / scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        lr = penalty * cls_up[CRowUp, CColUp] * cfg.TRACK.LR
        new_width = lr * ave_w + (1 - lr) * self.size[0]
        new_height = lr * ave_h + (1 - lr) * self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 0, img.shape[1])
        height = bbox_clip(new_height, 0, img.shape[0])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
        }

    def track(self,img):

        x_crop, scale_z = self.get_x_crop(img)
        output_dict = self.x_crop_2_res(img, x_crop, scale_z)
        return output_dict

    def track_adv(self, img, GAN, save_path=None, frame_id=None ):

        x_crop, scale_z = self.get_x_crop(img)
        '''Adversarial Attack'''
        x_crop_adv = adv_attack_search_gat(x_crop, GAN)
        '''predict'''
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path != None and frame_id != None:
            '''save'''
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def init_adv(self, img, bbox, GAN, save_path=None, name=None):

        z_crop,roi = self.get_z_crop(img, bbox)
        '''Adversarial Attack'''
        z_crop_adv = adv_attack_template(z_crop, GAN)
        self.model.template_gat(z_crop_adv,roi)
        '''save'''
        if save_path != None and name != None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name+'_clean.jpg'),z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'), z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def init_adv_S(self, img, bbox, GAN, save_path=None, name=None):
        z_crop,roi = self.get_z_crop(img, bbox)
        '''Adversarial Attack'''
        z_crop_adv = adv_attack_template_S(z_crop, GAN)
        self.model.template_gat(z_crop_adv,roi)
        '''save'''
        if save_path != None and name != None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name+'_clean.jpg'),z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'), z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def track_gauss(self, img, sigma, save_path=None, frame_id=None):
        x_crop, scale_z = self.get_x_crop(img)
        '''Gaussian Attack'''
        x_crop_adv = add_gauss_noise(x_crop, sigma)
        '''predict'''
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path != None and frame_id != None:
            '''save'''
            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_impulse(self, img, prob, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        '''impulse Attack'''
        x_crop_adv = add_pulse_noise(x_crop, prob)
        '''predict'''
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        if save_path != None and frame_id != None:
            '''save'''

            self.save_img(x_crop, x_crop_adv, save_path, frame_id)
        return output_dict

    def track_heatmap(self, img):
        x_crop, scale_z = self.get_x_crop(img)
        output_dict = self.x_crop_2_res(img, x_crop, scale_z)
        '''process score map'''
        score_map = np.max(self.score.reshape(5, 25, 25), axis=0)
        return output_dict, score_map

    '''supplementary material'''

    def track_supp(self, img, GAN, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        '''save clean region and heatmap'''
        x_crop_img = tensor2img(x_crop)
        cv2.imwrite(os.path.join(save_path, 'ori_search_%d.jpg' % frame_id), x_crop_img)
        '''original heatmap'''
        outputs_clean = self.model.track(x_crop)
        score = self._convert_score(outputs_clean['cls'])  # (25x25x5,)
        heatmap_clean = 255.0 * np.max(score.reshape(5, 25, 25), axis=0)  # [0,1]
        heatmap_clean = cv2.resize(heatmap_clean, (255, 255), interpolation=cv2.INTER_CUBIC)
        heatmap_clean = cv2.applyColorMap(heatmap_clean.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_clean_%d.jpg' % frame_id), heatmap_clean)
        '''Adversarial Attack'''
        x_crop_adv = adv_attack_search_gat(x_crop, GAN)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        '''save adv region and heatmap'''
        x_crop_img_adv = tensor2img(x_crop_adv)
        cv2.imwrite(os.path.join(save_path, 'adv_search_%d.jpg' % frame_id), x_crop_img_adv)
        score_adv = self.score
        heatmap_adv = 255.0 * np.max(score_adv.reshape(5, 25, 25), axis=0)  # [0,1]
        heatmap_adv = cv2.resize(heatmap_adv, (255, 255), interpolation=cv2.INTER_CUBIC)
        heatmap_adv = cv2.applyColorMap(heatmap_adv.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_adv_%d.jpg' % frame_id), heatmap_adv)
        return output_dict

    def save_img(self, tensor_clean, tensor_adv, save_path, frame_id):
        # print(frame_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ## clean x_crop
        img_clean = tensor2img(tensor_clean)
        cv2.imwrite(os.path.join(save_path, '%04d_clean.jpg' % frame_id), img_clean)
        ## adv x_crop
        img_adv = tensor2img(tensor_adv)
        cv2.imwrite(os.path.join(save_path, '%04d_adv.jpg' % frame_id), img_adv)
        ## diff
        tensor_diff = (tensor_adv - tensor_clean)
        tensor_diff += torch.abs(torch.min(tensor_diff))
        tensor_diff /= torch.max(tensor_diff)
        # print(torch.mean(torch.abs(tensor_diff)))
        tensor_diff *= 255
        img_diff = tensor2img(tensor_diff)
        cv2.imwrite(os.path.join(save_path, '%04d_diff.jpg' % frame_id), img_diff)