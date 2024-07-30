import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from models.inr import make_coord
import cv2

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=False,
                light_idx=-1, max_h=512, max_w=640, reverse=False, both=False, fix_range=False, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth
        self.light_idx=light_idx
        self.max_h=max_h
        self.max_w=max_w
        self.reverse = reverse
        self.both = both
        self.fix_range = fix_range
        
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    if self.light_idx == -1:
                        for light_idx in range(7):
                            if self.both:
                                metas.append((scan, light_idx, ref_view, src_views, 1)) # add 1, 0 for reverse depth
                            metas.append((scan, light_idx, ref_view, src_views, 0))
                    else:
                        if self.both:
                            metas.append((scan, self.light_idx, ref_view, src_views, 1))
                        metas.append((scan, self.light_idx, ref_view, src_views, 0))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # img = cv2.imread(filename)
        np_img = np.array(img)
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views, flip_flag = meta
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
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))

            
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            if i == 0:# just use in return
                depth_name = depth_filename
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)
            extrinsics_list.append(extrinsics)
            intrinsics_list.append(intrinsics)

            if i == 0:  # reference view
                if not self.fix_range:
                    depth_end = depth_interval * (self.ndepths-1) + depth_min # sample: 0:n-1
                else:
                    depth_end = 935 # pre-defined in DTU dataset
                if self.inverse_depth: #slice inverse depth
                    depth_max = depth_interval * self.ndepths + depth_min
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_max, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)
                else:
                    depth_values = np.linspace(depth_min, depth_end, self.ndepths)
                    depth_values = depth_values.astype(np.float32)
                    #depth_values = np.concatenate((depth_values,depth_values[::-1]),axis=0)
                    depth_max = depth_interval * self.ndepths + depth_min
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)

                depth = self.read_depth(depth_filename)
                mask = np.array((depth >= depth_min+depth_interval) & (depth <= depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)

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
