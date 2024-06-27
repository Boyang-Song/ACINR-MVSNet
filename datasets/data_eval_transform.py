import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *

from datasets.preprocess import *
from models.inr import make_coord
# Test any dataset with scale and center crop

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=True,
                light_idx=3, adaptive_scaling=True, max_h=960, max_w=1280, base_image_size=64, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth
        self.light_idx=light_idx
        self.adaptive_scaling=adaptive_scaling
        self.max_h=max_h
        self.max_w=max_w
        self.base_image_size=base_image_size
        
        assert self.mode == "test"
        self.metas = self.build_list()
        print('Data Loader : ********** dtu_test **********' )

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            # pair_file = "{}/pair.txt".format(scan)
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, self.light_idx, ref_view, src_views))
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
        scan, light_idx, ref_view, src_views = meta
        
        if self.nviews>len(src_views):
              self.nviews=len(src_views)+1
              
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        depth_bound = None
        proj_matrices = []
        cams=[]
        extrinsics_list=[]
        depth_params=[]

        for i, vid in enumerate(view_ids):
            # img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            # proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(vid)

            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            cams.append(intrinsics)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics_list.append(extrinsics)
            
            if i == 0:  # reference view
                if self.inverse_depth: #slice inverse depth
                    # print('Process {} inverse depth'.format(idx))
                    depth_max = depth_interval * self.ndepths + depth_min
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_max, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)
                else:
                    depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval ,
                                            dtype=np.float32) 
                                                                           
                    # depth_end = depth_interval * (self.ndepths-1) + depth_min # sample: 0:n-1
                    depth_max = depth_interval * self.ndepths + depth_min
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)

                depth_params.append(depth_min)
                depth_params.append(depth_interval)
                depth_params = np.array(depth_params).astype(np.float32)

        h_scale = float(self.max_h) / imgs[0].shape[0]
        w_scale = float(self.max_w) / imgs[0].shape[1]
        if h_scale > 1 or w_scale > 1:
            print("max_h, max_w should < W and H!")
            exit()
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
        scaled_input_images, scaled_input_cams= scale_dtu_input(imgs, cams, scale=resize_scale)

        # crop to fit network
        croped_imgs, croped_cams= crop_dtu_input(scaled_input_images, scaled_input_cams,
                                                               height=self.max_h, width=self.max_w,
                                                               base_image_size=self.base_image_size)
        ref_image = croped_imgs[0].copy()
####################################################################################################################
        original_image = ref_image  # [H, W, 3]
        original_image_h, original_image_w = original_image.shape[:2]
        original_image_hr = np.array(Image.fromarray(original_image).resize((original_image_w // 2, original_image_h // 2), Image.BICUBIC))
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

        hr_coord = make_coord(shape=[self.max_h // 2, self.max_w // 2], flatten=True)

        for i, image in enumerate(croped_imgs):
            image = image.astype(np.float32).transpose(2, 0, 1) / 255.
            image = (image - np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            image = torch.from_numpy(image).float()
            croped_imgs[i] = image.contiguous()
####################################################################################################################
        croped_imgs = torch.stack(croped_imgs)
        extrinsics_list = np.stack(extrinsics_list)
        intrinsics_list = np.stack(croped_cams)
        new_proj_matrices = []
        for id in range(self.nviews):
            ######
            croped_cams[id][:2, :] /= 4
            ######
            proj_mat = extrinsics_list[id].copy()
            proj_mat[:3, :4] = np.matmul(croped_cams[id], proj_mat[:3, :4])
            new_proj_matrices.append(proj_mat)

        new_proj_matrices = np.stack(new_proj_matrices)

        return {"imgs": croped_imgs,
                "proj_matrices": new_proj_matrices,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "extrinsics_list":extrinsics_list,
                "intrinsics_list":intrinsics_list,
                'image': image_hr,
                'hr_coord': hr_coord,
                "depth_params": depth_params,
                "ref_image": ref_image,
                "lr_image": lr_image,
                "depth_bound": depth_bound
                }
