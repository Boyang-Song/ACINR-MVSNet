import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

from datasets.preprocess import *
from models.inr import make_coord
import cv2
from collections import defaultdict

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, inverse_depth=False,
                max_h=576, max_w=768, reverse=False, both=False, image_scale=0.25, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.inverse_depth = inverse_depth
        self.max_h=max_h
        self.max_w=max_w
        self.reverse = reverse
        self.both = both
        self.image_scale = image_scale # use to resize in BlendedMVS
        
        assert self.mode in ["train", "val", "test"]
        self.metas, self.ref_views_per_scan= self.build_list()
        self.build_proj_mats()

    def build_list(self):
        metas = []
        ref_views_per_scan = defaultdict(list)
        with open(self.listfile) as f:
            self.scans = f.readlines()
            self.scans = [line.rstrip() for line in self.scans]

        # scans
        for scan in self.scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    ref_views_per_scan[scan] += [ref_view]
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) < self.nviews -1:
                        print('less ref_view small {}'.format(self.nviews-1))
                        continue
                    if self.both:
                        metas.append((scan, ref_view, src_views, 1)) # add 1, 0 for reverse depth
                    metas.append((scan, ref_view, src_views, 0))
        print("dataset", self.mode, "metas:", len(metas))
        return metas, ref_views_per_scan

    def __len__(self):
        return len(self.metas)

    def build_proj_mats(self):
        self.proj_mats = {} # proj mats for each scan
        self.depth_min = {}  # proj mats for each scan
        self.scale_factors = {} # depth scale factors for each scan
        for scan in self.scans:
            self.proj_mats[scan] = {}
            self.depth_min[scan] = {}
            for vid in self.ref_views_per_scan[scan]:
                proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
                intrinsics, extrinsics, depth_min = self.read_cam_file(scan, proj_mat_filename)
                self.proj_mats[scan][vid] = (intrinsics, extrinsics, depth_min)
                self.depth_min[scan][vid] = depth_min

    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        if self.image_scale != 1.0: # origin: 1.0
            intrinsics[:2, :] *= self.image_scale

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        if scan not in self.scale_factors:
            # use the first cam to determine scale factor
            self.scale_factors[scan] = 100/depth_min

        depth_min *= self.scale_factors[scan]
        extrinsics[:3, 3] *= self.scale_factors[scan]
        return intrinsics, extrinsics, depth_min

    def read_img(self, filename):
        img = Image.open(filename)
        # img = cv2.imread(filename)
        np_img = np.array(img)
        return np_img

    def read_depth(self, scan, filename):
        # read pfm depth file
        depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_image *= self.scale_factors[scan]
        return depth_image


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views, flip_flag = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        depth_bound = None
        proj_matrices = []
        extrinsics_list = []
        intrinsics_list = []
        depth_params = []
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 000000000
            img_filename = os.path.join(self.datapath,
                                        '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            # if i == 0:
            #     print('process in {}, {}'.format(idx, img_filename))

            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            
            if i == 0:
                depth_name = depth_filename
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min = self.proj_mats[scan][vid]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

            if i == 0:  # reference view
                if self.inverse_depth: #slice inverse depth
                    print('inverse depth')
                    depth = self.read_depth(scan, depth_filename)
                    depth_max = depth.max()

                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_max, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_interval = depth_values[1] - depth_values[0]
                    depth_values = depth_values.astype(np.float32)
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)

                else:
                    depth = self.read_depth(scan, depth_filename)
                    depth_max = depth.max()
                    depth_interval = (depth_max - depth_min) / self.ndepths

                    depth_end = depth_interval * (self.ndepths - 1) + depth_min  # sample: 0:n-1
                    depth_values = np.linspace(depth_min, depth_end, self.ndepths)
                    depth_values = depth_values.astype(np.float32)
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)

                mask = np.array((depth > depth_min+depth_interval) & (depth < depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)

                mask_depth = depth.copy()
                ds=depth_min+depth_interval
                de=depth_min+(self.ndepths-2)*depth_interval
                ret, mask_depth = cv2.threshold(mask_depth, ds, 100000, cv2.THRESH_TOZERO)
                ret, mask_depth = cv2.threshold(mask_depth, de, 100000, cv2.THRESH_TOZERO_INV)
                mask_depth = np.expand_dims(mask_depth, 2)
                mask_depth = torch.tensor(mask_depth).permute(2, 0, 1).float()

                depth_params.append(depth_min)
                depth_params.append(depth_interval)
                depth_params=np.array(depth_params).astype(np.float32)
####################################################################################################################
        original_image = imgs[0].copy()   # [H, W, 3]
        original_image_h, original_image_w = original_image.shape[:2]
        original_image_hr = np.array(Image.fromarray(original_image).resize((original_image_w // 4, original_image_h // 4), Image.BICUBIC))
        original_image_lr = np.array(Image.fromarray(original_image).resize((original_image_w // 4, original_image_h // 4), Image.BICUBIC))

        image_hr = original_image_hr.astype(np.float32).transpose(2, 0, 1) / 255.
        lr_image = original_image_lr.astype(np.float32).transpose(2, 0, 1) / 255.
        image_hr = (image_hr - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        lr_image = (lr_image - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        # to tensor
        image_hr = torch.from_numpy(image_hr).float()
        lr_image = torch.from_numpy(lr_image).float()
        image_hr = image_hr.contiguous()
        lr_image = lr_image.contiguous()

        hr_coord = make_coord(shape=[self.max_h // 4, self.max_w // 4], flatten=True)

        for i, image in enumerate(imgs):
            image = image.astype(np.float32).transpose(2, 0, 1) / 255.
            image = (image - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image = torch.from_numpy(image).float()
            imgs[i] = image.contiguous()
####################################################################################################################
        imgs = torch.stack(imgs)
        proj_matrices = np.stack(proj_matrices)
        extrinsics_list = np.stack(extrinsics_list)
        intrinsics_list = np.stack(intrinsics_list)

        if (flip_flag and self.both) or (self.reverse and not self.both):
            depth_values = np.array([depth_values[len(depth_values)-i-1]for i in range(len(depth_values))])

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "mask_depth": mask_depth,
                "depth_values": depth_values, # generate depth index
                "mask": mask,
                'name':depth_name,
                "extrinsics_list": extrinsics_list,
                "intrinsics_list": intrinsics_list,
                'image': image_hr,
                'hr_coord': hr_coord,
                "depth_params":depth_params,
                "lr_image": lr_image,
                "depth_bound": depth_bound
                }
