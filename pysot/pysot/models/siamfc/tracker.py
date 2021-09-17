from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker  # GOT-10k

from .utils import init_weights, crop_and_resize, read_image, show_image, load_pretrain
from .backbones import AlexNetV0, AlexNetV1, ResNet22, ResNeXt22, ResNet22W
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from .network import SiamFCNet
from .config import config
from tqdm import tqdm
from torch.utils.data.distributed import \
    DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from .utils import get_logger

__all__ = ['SiamFCTracker']


class SiamFCTracker(Tracker):

    def __init__(self, model_path=None, cfg=None, template_size=4, group=2):
        super(SiamFCTracker, self).__init__(model_path, True)

        self.cfg = config


        if cfg:
            config.update(cfg)
            # setup model


        if group == 2:
            self.net = SiamFCNet(backbone=AlexNetV1(), head=SiamFC(self.cfg.out_scale))

        else:
            self.net = SiamFCNet(backbone=AlexNetV0(), head=SiamFC(self.cfg.out_scale))

        init_weights(self.net)


        if model_path is not None:
            if 'alexnet' in model_path and group == 1:
                self.net = load_pretrain(self.net, model_path)  # load pretrain
            elif 'siamfc' in model_path:
                self.net.load_state_dict(torch.load(
                    model_path, map_location=lambda storage, loc: storage))

        # self.net = self.net.to(self.device)
        self.net = self.net.cuda()

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum)

        gamma = np.power(config.ultimate_lr / config.initial_lr, 1.0 / config.epoch_num)

        # gamma=0.87
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)



    @torch.no_grad()  #
    def init(self, img, box):
        # set to evaluation mode
        self.bbox = box
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        self.t_s_rate = float((self.target_sz[0] * self.target_sz[1]) / (self.center[0] * self.center[1]))
        # create hanning window  response_up=16 ；  response_sz=17 ； self.upscale_sz=272
        self.upscale_sz = config.response_up * config.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = config.scale_step ** np.linspace(
            -(config.scale_num // 2),
            config.scale_num // 2, config.scale_num)

        # exemplar and search sizes  config.context=1/2
        context = config.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * 1 * config.instance_sz / config.exemplar_sz

        self.avg_color = np.mean(img, axis=(0, 1))
        z = crop_and_resize(img, self.center, self.z_sz,
                            out_size=config.exemplar_sz,
                            border_value=self.avg_color)


        z = torch.from_numpy(z).cuda().permute(2, 0, 1).unsqueeze(0).float()  # [1,3,127,127]

        feature_z = self.net.features(z)  # [1,3,6,6]
        self.kernel = feature_z


    @torch.no_grad()  #
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=config.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float()  # [3,3,255,255]

        # responses
        x = self.net.features(x)

        responses = self.net.head(self.kernel, x)  # [3,1,17,17]

        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:config.scale_num // 2] *= config.scale_penalty
        responses[config.scale_num // 2 + 1:] *= config.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - config.window_influence) * response + \
                   config.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           config.total_stride / config.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / config.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - config.scale_lr) * 1.0 + config.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box  [x,y,w,h]
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def get_z_crop(self, img, bbox):
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
        return z_crop
    
    def init_adv(self, img, bbox, GAN, save_path=None, name=None):
        z_crop = self.get_z_crop(img, bbox)
        '''Adversarial Attack'''
        z_crop_adv = adv_attack_template(z_crop, GAN)
        self.model.template(z_crop_adv)
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
        z_crop = self.get_z_crop(img, bbox)
        '''Adversarial Attack'''
        z_crop_adv = adv_attack_template_S(z_crop, GAN)
        self.model.template(z_crop_adv)
        '''save'''
        if save_path != None and name != None:
            z_crop_img = tensor2img(z_crop)
            cv2.imwrite(os.path.join(save_path, name+'_clean.jpg'),z_crop_img)
            z_crop_adv_img = tensor2img(z_crop_adv)
            cv2.imwrite(os.path.join(save_path, name + '_adv.jpg'), z_crop_adv_img)
            diff = z_crop_adv - z_crop
            diff_img = tensor2img(diff)
            cv2.imwrite(os.path.join(save_path, name + '_diff.jpg'), diff_img)

    def track(self, img_files, box, visualize=False):  # x,y,w,h
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        if len(box) == 8:  # vot
            box = self.get_axis_aligned_bbox(box)

        boxes[0] = box
        times = np.zeros(frame_num)
        num = 0
        for f, img_file in enumerate(img_files):
            num += 1
            img = read_image(img_file)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :], delay=1)
        return boxes, times

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)  # 训练模式

        # parse batch data
        z = batch[0].cuda()  # to(self.device, non_blocking=self.cuda)
        x = batch[1].cuda()  # to(self.device, non_blocking=self.cuda)

        z_x_rate = batch[2]
        rates = []
        for r in z_x_rate:
            if r >= 0.008:
                rate = 6

                rates.append(rate)
            # elif r < 0.005:
            #     rate = 2
            #     rates.append(rate)
            else:
                rate = 4

                rates.append(rate)

        with torch.set_grad_enabled(backward):

            # inference
            responses = self.net(z, x, rates)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    # 在禁止计算梯度下调用被允许计算梯度的函数
    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None, save_dir='models'):
        # set to train mode

        logger = get_logger('./models/logs/train_log.log')
        logger.info('start training!')
        self.net.train()
        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=config.exemplar_sz,  # 127
            instance_sz=config.instance_sz,  # 255
            context=config.context)  #

        dataset = Pair(seqs=seqs, transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True)

        # loop over epochs
        for epoch in range(config.resume_epoch, config.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in tqdm(enumerate(dataloader)):
                loss = self.train_step(batch, backward=True)

                # print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(epoch + 1, it + 1, len(dataloader), loss))
                logger.info('Epoch: {} [{}/{}] Loss: {:.5f} '.format(epoch + 1, it + 1, len(dataloader), loss))

                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            model_path = os.path.join(save_dir, 'siamfc_%d.pth' % (epoch + 1))

            if torch.cuda.device_count() > 1:  # 多GPU

                torch.save(self.net.module.state_dict(), model_path)

            else:  # 单GPU
                torch.save(self.net.state_dict(), model_path)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance

            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg, np.ones_like(x) * 0.5, np.zeros_like(x)))

            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = config.r_pos / config.total_stride
        r_neg = config.r_neg / config.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).cuda().float()

        return self.labels

    def get_axis_aligned_bbox(self, region):
        """ convert region to (cx, cy, w, h) that represent by axis aligned box
        """

        nv = region.size
        if nv == 8:
            cx = np.mean(region[0::2])
            cy = np.mean(region[1::2])
            x1 = min(region[0::2])
            x2 = max(region[0::2])
            y1 = min(region[1::2])
            y2 = max(region[1::2])
            A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
                 np.linalg.norm(region[2:4] - region[4:6])
            A2 = (x2 - x1) * (y2 - y1)
            s = np.sqrt(A1 / A2)
            w = s * (x2 - x1) + 1
            h = s * (y2 - y1) + 1
        else:
            x = region[0]
            y = region[1]
            w = region[2]
            h = region[3]
            cx = x + w / 2
            cy = y + h / 2

        box = [cx, cy, w, h]
        return box
