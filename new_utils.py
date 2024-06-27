import logging
import os
import sys
import time
from collections import defaultdict
from collections import deque
import torch
import random

import numpy as np
import os.path as osp
import cv2
from torchvision import transforms

# def seed_everything(seed):
#     seed=int(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


def setup_logger(name, save_dir, prefix="", timestamp=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        timestamp = time.strftime(".%m_%d_%H_%M_%S") if timestamp else ""
        prefix = "." + prefix if prefix else ""
        log_file = os.path.join(save_dir, "log{}.txt".format(prefix + timestamp))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger

class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        if np.sum(self.counts) == 0:
            return 0
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            count = 1
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    count = v.numel()
                    v = v.sum().item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v, count)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return object.__getattr__(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.avg, meter.global_avg)
            )
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(metric_str)


def eval_file_logger(sample, outputs, save_dir, filename, scene_name_index=-3, out_index_minus=0):
    ndepth=96
    l = filename.split("/")

    scene = l[scene_name_index]

    scene_folder = osp.join(save_dir, "depthfusion", scene)

    if not osp.isdir(scene_folder):
        mkdir(scene_folder)
        print("**** {} ****".format(scene))

    out_index = int(l[-1][5:8]) - out_index_minus

    ref_cam_paras=np.zeros((2,4,4), dtype=np.float32)
    extrinsics_list = sample["extrinsics_list"].detach().cpu().numpy().copy()
    intrinsics_list = sample["intrinsics_list"].detach().cpu().numpy().copy()
    depth_params = sample["depth_params"].detach().cpu().numpy().copy()

    for i in range(0, 4):
        for j in range(0, 4):
            ref_cam_paras[0,i,j]=extrinsics_list[0,0,i,j]
    for i in range(0, 3):
        for j in range(0, 3):
            ref_cam_paras[1,i,j]=intrinsics_list[0,0,i,j]
    ref_cam_paras[1,3,0]=depth_params[0,0]
    ref_cam_paras[1,3,1]=depth_params[0,1]
    ref_cam_paras[1,3,2] = ndepth
    ref_cam_paras[1,3,3] = ref_cam_paras[1,3,0]+(ndepth-1)*ref_cam_paras[1,3,1]

    out_ref_image_path = scene_folder + ('/%08d.jpg' % out_index)
    ref_image = sample["ref_image"][0].detach().cpu().numpy().copy()
    # cv2.imwrite(out_ref_image_path, ref_image)
    transforms.ToPILImage()(ref_image).save(out_ref_image_path)

    init_depth_map_path = scene_folder + ('/%08d_init.pfm' % out_index)
    init_prob_map_path = scene_folder + ('/%08d_init_prob.pfm' % out_index)

    init_depth_map = outputs["coarse_depth"].copy()[0, 0]
    init_prob_map = outputs["photometric_confidence"].copy()[0, 0]

    write_pfm(init_depth_map_path, init_depth_map)
    write_pfm(init_prob_map_path, init_prob_map)

    out_init_cam_path = scene_folder + ('/cam_%08d_init.txt' % out_index)
    init_cam_paras = ref_cam_paras.copy()
    init_cam_paras[1, :2, :3] *= (float(init_depth_map.shape[0]) / ref_image.shape[0])
    write_cam_dtu(out_init_cam_path, init_cam_paras)

    for i, k in enumerate(outputs.keys()):
        if "flow" in k:
            out_flow_depth_map = outputs[k].copy()[0, 0]
            flow_depth_map_path = scene_folder + "/{:08d}_{}.pfm".format(out_index, k)
            write_pfm(flow_depth_map_path, out_flow_depth_map)
            out_flow_cam_path = scene_folder + "/cam_{:08d}_{}.txt".format(out_index, k)
            flow_cam_paras = ref_cam_paras.copy()
            flow_cam_paras[1, :2, :3] *= (float(out_flow_depth_map.shape[0]) / float(ref_image.shape[0]))
            write_cam_dtu(out_flow_cam_path, flow_cam_paras)


def mkdir(path):
    os.makedirs(path, exist_ok=True)

def write_cam_dtu(file, cam):
    # f = open(file, "w")
    f = open(file, "w")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write(
        '\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def write_pfm(file, image, scale=1):
    file = open(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()