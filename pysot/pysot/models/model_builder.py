# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import *
import os
from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car import CARHead
from pysot.models.neck import get_neck
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head, get_ban_head

from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output

class ModelBuilder(nn.Module):
    def __init__(self,tracker):
        super(ModelBuilder, self).__init__()
        self.tracker = tracker
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        print(self.tracker)
        if 'gat' not in self.tracker:
            # build adjust layer
            if cfg.ADJUST.ADJUST:
                self.neck = get_neck(cfg.ADJUST.TYPE,
                                     **cfg.ADJUST.KWARGS)
        if 'car' in self.tracker:
            # build car head
            self.car_head = CARHead(cfg, 256)
            self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

        elif 'rpn' in self.tracker:
            # build rpn head
            self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                         **cfg.RPN.KWARGS)
        elif 'mask' in self.tracker:
            # build mask head
            if cfg.MASK.MASK:
                self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                               **cfg.MASK.KWARGS)
                if cfg.REFINE.REFINE:
                    self.refine_head = get_refine_head(cfg.REFINE.TYPE)

        elif 'gat' in self.tracker:
            # build car head
            self.car_head = CARHead(cfg, 256)
            # build response map
            self.attention = Graph_Attention_Union(256, 256)

        elif 'ban' in self.tracker:
            # build ban head
            if cfg.BAN.BAN:
                self.head = get_ban_head(cfg.BAN.TYPE,
                                         **cfg.BAN.KWARGS)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)


    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        # return self.zf

    def template_gat(self,z,roi):
        zf = self.backbone(z, roi)
        self.zf = zf
        return self.zf

    def search(self,x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)  #<list>
        return xf

    def get_three_features(self, x):
        '''

            Args:
                x: search image

            Returns:
                features cnn by x and z
            '''

        xf = self.search(x)

        features = self.xcorr_depthwise(xf[0], self.zf[0])  # [N,256,W,H]

        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, features_new], 1)

        return features


    def get_features(self,x):

        feature = self.get_three_features(x)
        feature = self.down(feature)

        return feature

    def get_attention(self, x):
        xf = self.backbone(x)
        attention_features = self.attention(self.zf, xf)
        # for i in range(attention_features.shape[1]):
        #     self.tensorshow(attention_features[0][i],'attention_feature_{}'.format(i))

        return attention_features

    def track(self, x):
        if 'car' in self.tracker:
            features = self.get_features(x)
            cls, loc, cen = self.car_head(features)
            return {
                'cls': cls,
                'loc': loc,
                'cen': cen,
                'features':features
               }
        if 'gat' in self.tracker:
            features = self.get_attention(x)
            cls, loc, cen = self.car_head(features)
            return {
                'cls': cls,
                'loc': loc,
                'cen': cen
            }
        if 'ban' in self.tracker:
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)
            cls, loc = self.head(self.zf, xf)
            return {
                'cls': cls,
                'loc': loc
            }
        else:
            xf = self.search(x)
            cls, loc = self.rpn_head(self.zf, xf)
            if cfg.MASK.MASK:
                mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
            return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
            }




    def log_softmax(self, cls):
        if 'ban' in self.tracker:
            if cfg.BAN.BAN:
                cls = cls.permute(0, 2, 3, 1).contiguous()
                cls = F.log_softmax(cls, dim=3)
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        if 'siamcar' in self.tracker:
            features = self.xcorr_depthwise(xf[0], zf[0])
            for i in range(len(xf)-1):
                features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
                features = torch.cat([features,features_new],1)
            features = self.down(features)

            cls, loc, cen = self.car_head(features)
            locations = compute_locations(cls, cfg.TRACK.STRIDE)
            cls = self.log_softmax(cls)
            cls_loss, loc_loss, cen_loss = self.loss_evaluator(
                locations,
                cls,
                loc,
                cen, label_cls, label_loc
                )
        if 'ban' in self.tracker:
            cls, loc = self.head(zf, xf)
            # cls loss with cross entropy loss
            cls = self.log_softmax(cls)
            cls_loss = select_cross_entropy_loss(cls, label_cls)
            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc, label_cls)

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['clc'], outputs['loc'], outputs['cen'] = cls, loc, cen
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs

