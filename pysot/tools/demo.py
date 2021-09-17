from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.model_load import load_pretrain
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, default='/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/model.pth', help='model name')
parser.add_argument('--tracker', type=str, default='siamcar_otb', help='tracker name')
parser.add_argument('--save_path', type=str, default='/home/marq/Desktop/CSA/pysot/clean_adv_imgs', help='model name')
parser.add_argument('--video_name', default='/home/marq/Desktop/datasets/OTB2015/Basketball/img', type=str,
                    help='videos or image files')
args = parser.parse_args()
print(args.tracker)
from GAN_utils_search_0 import *
from common_path import *
model_name = opt.model

if args.tracker == 'siamcar_otb':
    args.config = '/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/config.yaml'
    args.snapshot = '/home/marq/Desktop/CSA/pysot/experiments/siamcar_r50/model.pth'
elif args.tracker == 'siamrpn++_otb':
    args.config = '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/config.yaml'
    args.snapshot = '/home/marq/Desktop/CSA/pysot/experiments/siamrpn_r50_l234_dwxcorr_otb/model.pth'
elif args.tracker == 'siamgat_otb':
    args.config = '/home/marq/Desktop/CSA/pysot/experiments/siamgat_googlenet/config.yaml'
    args.snapshot = '/home/marq/Desktop/CSA/pysot/experiments/siamgat_googlenet/model.pth'
else:
    assert error

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    if 'siamcar_otb' in args.tracker:
        params = getattr(cfg.HP_SEARCH, 'OTB2015')
        hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder(args.tracker)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    for frame in get_frames(args.video_name):
        if first_frame:
            fram = 1
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)
                init_rect = [198,214,34,81] #Basketball_260
               # init_rect = [250,168,106,105] #blurcar1_600
               # init_rect = [164,121,27,24] #crossing_70
               # init_rect = [336,165,26,61] #mountainbike
               # init_rect = [57,156,198,173] #mountainbike
               # init_rect = [136,35,52,182] #mountainbike
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            fram += 1
            print(fram)
            # assert fram < 5

            if 'siamcar_otb' in args.tracker:
                outputs = tracker.track(frame, hp)
                # outputs = tracker.track_adv(frame, GAN_1, hp, save_path=args.save_path,frame_id=fram)
            else:
                outputs = tracker.track(frame)
                # outputs = tracker.track_adv(frame, GAN_1, save_path=args.save_path, frame_id=fram)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
                print(bbox)
            cv2.imshow(video_name, frame)
            # if 479 < fram < 500:
            #     cv2.waitKey(3000)
            cv2.waitKey(40)


if __name__ == '__main__':
    main()
