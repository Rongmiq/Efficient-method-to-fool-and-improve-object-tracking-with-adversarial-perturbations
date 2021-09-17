# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str

from common_path import *
'''GAN'''
from GAN_utils_template import *
model_name = opt.model

parser = argparse.ArgumentParser(description='siamcar_r50 tracking')
parser.add_argument('--tracker_name',default='siamcar_r50',type=str)
parser.add_argument('--tracker',default='siamcar_otb',type=str)
parser.add_argument('--dataset', default= 'VOT2018', type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False, action='store_true',
        help='whether visualzie result')
parser.add_argument('--adv', default=True, action='store_true',
        help='whether adv')
parser.add_argument('--save_path', type=str, default=None,
        help='config file')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml',
        help='config file')
parser.add_argument('--model_name', type=str, default='latest_net_G.pth',
        help='choose the .pth ')
args = parser.parse_args()

torch.set_num_threads(1)

def main():

    snapshot_path = os.path.join(project_path_, 'pysot/experiments/%s/model.pth' % args.tracker_name)
    config_path = os.path.join(project_path_, 'pysot/experiments/%s/config.yaml' % args.tracker_name)
    # load config
    cfg.merge_from_file(config_path)

    # hp_search
    if 'car' in args.tracker_name:
        params = getattr(cfg.HP_SEARCH, args.dataset)
        hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}
    dataset_root = os.path.join(dataset_root_, args.dataset)
    # create model
    '''a model is a Neural Network.(a torch.nn.Module)'''
    model = ModelBuilder(tracker=args.tracker)

    # load model
    model = load_pretrain(model, snapshot_path).cuda().eval()

    # build tracker
    '''a tracker is a object, which consists of not only a NN but also some post-processing'''
    tracker = build_tracker(model)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # save_img_path = os.path.join(args.save_path,model_name,args.tracker_name)
    # print('img have saved in',save_img_path)
    # if not os.path.isdir(save_img_path):
    #     os.makedirs(save_img_path)
    save_img_path = None
    if args.adv:
        flag = 'att'
    else:
        flag = 'no_att'
    # model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            video_path = os.path.join('results', args.dataset, args.tracker_name+str(args.adv),
                                      'baseline', video.name)
            if os.path.exists(video_path):
                print(video_path,'exists, skipping')
            else:

                if args.video != '':
                    # test one special video
                    if video.name != args.video:
                        continue
                frame_counter = 0
                lost_number = 0
                toc = 0
                pred_bboxes = []
                for idx, (img, gt_bbox) in enumerate(video):
                    if len(gt_bbox) == 4:
                        gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                           gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                           gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                    tic = cv2.getTickCount()
                    if idx == frame_counter:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                        '''GAN'''
                        if flag == 'att':
                            tracker.init_adv(img, gt_bbox_,GAN,save_path=None,name=video.name)
                        else:
                            tracker.init(img, gt_bbox_)
                        pred_bbox = gt_bbox_
                        pred_bboxes.append(1)
                    elif idx > frame_counter:
                        if 'car' in args.tracker_name:
                            outputs = tracker.track(img,hp)
                        else:
                            outputs = tracker.track(img)
                        pred_bbox = outputs['bbox']
                        if ('siamgat' not in args.tracker_name) and ('siamban' not in args.tracker_name):
                            if cfg.MASK.MASK:
                                pred_bbox = outputs['polygon']
                        overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                        if overlap > 0:
                            # not lost
                            pred_bboxes.append(pred_bbox)
                        else:
                            # lost object
                            pred_bboxes.append(2)
                            frame_counter = idx + 5 # skip 5 frames
                            lost_number += 1
                    else:
                        pred_bboxes.append(0)
                    toc += cv2.getTickCount() - tic
                    if idx == 0:
                        cv2.destroyAllWindows()
                    if args.vis and idx > frame_counter:
                        cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 0), 3)
                        if cfg.MASK.MASK:
                            cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                    True, (0, 255, 255), 3)
                        else:
                            bbox = list(map(int, pred_bbox))
                            cv2.rectangle(img, (bbox[0], bbox[1]),
                                          (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(video.name, img)
                        cv2.waitKey(1)
                toc /= cv2.getTickFrequency()
                # save results
                video_path = os.path.join('results', args.dataset, args.tracker_name+str(args.adv),
                        'baseline', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        if isinstance(x, int):
                            f.write("{:d}\n".format(x))
                        else:
                            f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
                print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                        v_idx+1, video.name, toc, idx / toc, lost_number))
                total_lost += lost_number
            print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            model_path = os.path.join('results', args.dataset, model_name + args.tracker_name+str(args.adv)[0])
            result = os.path.join(model_path, '{}.txt'.format(video.name))
            if os.path.exists(result):
                print(result,'exists, skipping')
            else:
                if args.video != '':
                    # test one special video
                    if video.name != args.video:
                        continue
                toc = 0
                pred_bboxes = []
                scores = []
                track_times = []
                for idx, (img, gt_bbox) in enumerate(video):
                    tic = cv2.getTickCount()
                    if idx == 0:
                        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                        '''GAN'''
                        if flag == 'att':
                            tracker.init_adv(img, gt_bbox_, GAN, save_path=None, name=video.name)
                        else:
                            tracker.init(img, gt_bbox_)
                        pred_bbox = gt_bbox_
                        scores.append(None)
                        if 'VOT2018-LT' == args.dataset:
                            pred_bboxes.append([1])
                        else:
                            pred_bboxes.append(pred_bbox)
                    else:
                        if 'car' in args.tracker_name:
                            outputs = tracker.track(img,hp)
                        else:
                            outputs = tracker.track(img)
                        pred_bbox = outputs['bbox']
                        pred_bboxes.append(pred_bbox)
                        # scores.append(outputs['best_score'])
                    toc += cv2.getTickCount() - tic
                    track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                    if idx == 0:
                        cv2.destroyAllWindows()
                    if args.vis and idx > 0:
                        gt_bbox = list(map(int, gt_bbox))
                        pred_bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                      (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                      (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                        cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.imshow(video.name, img)
                        cv2.waitKey(1)
                toc /= cv2.getTickFrequency()
                # save results
                if 'VOT2018-LT' == args.dataset:
                    video_path = os.path.join('results', args.dataset, model_name,
                            'longterm', video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path,
                            '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                    result_path = os.path.join(video_path,
                            '{}_001_confidence.value'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in scores:
                            f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                    result_path = os.path.join(video_path,
                            '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                elif 'GOT-10k' == args.dataset:
                    video_path = os.path.join('results', args.dataset, model_name, video.name)
                    if not os.path.isdir(video_path):
                        os.makedirs(video_path)
                    result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                    result_path = os.path.join(video_path,
                            '{}_time.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in track_times:
                            f.write("{:.6f}\n".format(x))
                else:
                    model_path = os.path.join('results', args.dataset, model_name + args.tracker_name+str(args.adv)[0])
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                    with open(result_path, 'w') as f:
                        for x in pred_bboxes:
                            f.write(','.join([str(i) for i in x])+'\n')
                print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                    v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
