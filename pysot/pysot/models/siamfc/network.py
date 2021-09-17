from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import os

__all__ = ['SiamFCNet']


class SiamFCNet(nn.Module):

    def __init__(self, backbone, head):
        super(SiamFCNet, self).__init__()
        self.features = backbone
        self.head = head

    def forward(self, z, x):
        feature_z = self.features(z)  # [8, 256, 6, 6]

        x = self.features(x)  # [8, 256, 20, 20]
        out = self.head(feature_z, x)  # [8, 1, 15, 15]
        return out

