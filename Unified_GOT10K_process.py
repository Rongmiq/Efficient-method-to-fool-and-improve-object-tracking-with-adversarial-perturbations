import os
import glob
import numpy as np
import cv2
from pysot.core.config import cfg
from siamrpnpp_utils import get_subwindow_numpy

'''sampling interval'''
interval = 10 # as the paper uses

def crop_z(img, bbox, channel_average):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                bbox[1] + (bbox[3] - 1) / 2])
    size = np.array([bbox[2], bbox[3]])

    # calculate z crop size
    w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    s_z = round(np.sqrt(w_z * h_z))

    # get crop with s_z
    z_crop = get_subwindow_numpy(img, center_pos,
                                cfg.TRACK.EXEMPLAR_SIZE,
                                s_z, channel_average)
    return z_crop
def crop_x(img, bbox, channel_average):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                bbox[1] + (bbox[3] - 1) / 2])
    size = np.array([bbox[2], bbox[3]])
    w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
    s_z = np.sqrt(w_z * h_z)
    '''s_x/s_z = cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE'''
    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

    # get crop with s_x
    x_crop = get_subwindow_numpy(img, center_pos,
                                 cfg.TRACK.INSTANCE_SIZE,
                                 round(s_x), channel_average)
    return x_crop

# if __name__ == '__main__':
    # '''change following two paths to yours!'''
    # # SSD is preferred for higher speed :)
    # got10k_path = '/home/xmq/Desktop/dataset/GOT10k'
    # save_path = '/home/xmq/Desktop/XiaoMa/GOT10K_reproduce'
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # train_path = os.path.join(got10k_path,'train')
    # video_list = sorted(os.listdir(train_path))
    # video_list.remove('list.txt')
    # num_video = len(video_list)
    # init_arr = np.zeros((num_video,4))
    # for i ,video in enumerate(video_list):
    #     save_dir = os.path.join(save_path,video)
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     frames_list = sorted(glob.glob(os.path.join(train_path,video,'*.jpg')))
    #     gt_file = os.path.join(train_path,video,'groundtruth.txt')
    #     gt_arr = np.loadtxt(gt_file,dtype=np.float64,delimiter=',')
    #     '''merge init gt into one file'''
    #     init_arr[i] = gt_arr[0].copy()
    #     init_frame = cv2.imread(frames_list[0])
    #     channel_average = np.mean(init_frame,axis=(0,1))
    #     '''crop & save initial template region'''
    #     z_crop = crop_z(init_frame, gt_arr[0], channel_average)
    #     dest_path = frames_list[0].replace(train_path, save_path)
    #     cv2.imwrite(dest_path, z_crop)
    #     '''crop search region every interval frames'''
    #     num_frames = len(frames_list)
    #     index_list = list(range(num_frames))[1:num_frames:interval]
    #     for index in index_list:
    #         frame_path = frames_list[index]
    #         cur_frame = cv2.imread(frame_path)
    #         x_crop = crop_x(cur_frame, gt_arr[index-1],channel_average)
    #         dest_path = frames_list[index].replace(train_path,save_path)
    #         cv2.imwrite(dest_path,x_crop)
    #     print('%d/%d completed.'%(i+1,num_video))
    # save_file = os.path.join(save_path,'init_gt.txt')
    # np.savetxt(save_file,init_arr,fmt='%.4f',delimiter=',')

if __name__ == '__main__':
    # def gt2roi(gt):
    #     bbox = []
    #     for i in range(len(gt)):
    #         bbox.append(float(gt[i]))
    #     print(bbox)
    #     # channel_average = np.mean(img, axis=(0, 1))
    #     size_w = bbox[2]
    #     size_h = bbox[3]
    #     p_w_h = size_w/size_h
    #
    #     w_z = size_w + 0.5 * (size_w + size_h)
    #     h_z = size_h + 0.5 * (size_w + size_h)
    #     s_z = w_z * h_z
    #
    #     p = size_w * size_h / s_z
    #
    #     crop_size = p * 127 * 127
    #     w = np.sqrt(crop_size * p_w_h)
    #     h = crop_size/w
    #
    #     box = []
    #     box.append(int(((127-w)/2)))
    #     box.append(int(((127+h)/2)))
    #     box.append(int(w))
    #     box.append(int(h))
    #     print(box)
    #     return box
    #
    # gth_path = os.path.join(r'/home/marq/Desktop/CSA/GOT10K_reproduce/init_gt.txt')
    # fp = open(gth_path)
    #
    # boxes = []
    #
    # for line in fp:
    #     box = []
    #     line = line.split('\n')
    #     # print(line[0])
    #     split_index = []
    #
    #     for i, x in enumerate(line[0]):
    #         if x == ',':
    #             split_index.append(i)
    #     box.append(line[0][:split_index[0]])
    #     box.append(line[0][split_index[0]+1 : split_index[1]])
    #     box.append(line[0][split_index[1]+1 : split_index[2]])
    #     box.append(line[0][split_index[2]+1 : ])
    #     box_gt = gt2roi(box)
    #
    #     # Write your code here
    #
    #     f = open("/home/marq/Desktop/CSA/GOT10K_reproduce/gt.txt", "a")
    #     f.write("{},{},{},{}\n".format(box_gt[0],box_gt[1],box_gt[2],box_gt[3]))
    #     f.close()

    fp = open(r'/home/marq/Desktop/CSA/g.txt')
    for line in fp:
        line = line.split('\n')
        box = np.array([0,0,0,0])

        split_index = []
        for i, x in enumerate(line[0]):
            if x == ',':
                split_index.append(i)
        box[0] = (line[0][:split_index[0]])
        box[1] = (line[0][split_index[0]+1 : split_index[1]])
        box[2] = (line[0][split_index[1]+1 : split_index[2]])
        box[3] = (line[0][split_index[2]+1 : ])
        print(box)
        print(box.shape)

