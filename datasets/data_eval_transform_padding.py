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
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, inverse_depth=True,
                max_h=1200, max_w=1600, base_image_size=8, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.inverse_depth = inverse_depth

        self.max_h = max_h
        self.max_w = max_w
        self.base_image_size = base_image_size
        self.depth_range = {'Family': (0.158, 8.465),
                            'Francis': (0.314, 17.959),
                            'Horse': (0.082, 5.503),
                            'Lighthouse': (0.229, 9.568),
                            'M60': (0.260, 6.501),
                            'Panther': (0.288, 6.860),
                            'Playground': (0.249, 7.491),
                            'Train': (0.165, 9.446)}
        assert self.mode == "test"
        self.metas = self.build_list()
        print('Data Loader : data_eval_transform_padding **************')

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))
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
        depth_interval = float(lines[11].split()[1])
        depth_end = float(lines[11].split()[3])
        return intrinsics, extrinsics, depth_min, depth_end

    def read_img(self, filename):
        img = Image.open(filename)
        # img = cv2.imread(filename)
        # # scale 0~255 to 0~1
        # np_img = np.array(img, dtype=np.float32)
        np_img = np.array(img)
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta

        if self.nviews > len(src_views):
            self.nviews = len(src_views) + 1

        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        depth_bound = None
        proj_matrices = []
        cams = []
        extrinsics_list = []
        depth_params = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(proj_mat_filename)
            cams.append(intrinsics)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics_list.append(extrinsics)

            if i == 0:  # reference view
                if self.inverse_depth:  # slice inverse depth
                    # print('inverse depth')
                    depth_values = np.linspace(1.0 / depth_min, 1.0 / depth_max, self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_interval = depth_values[1] - depth_values[0]
                    depth_values = depth_values.astype(np.float32)
                    depth_bound = np.linspace(depth_min, depth_max, self.ndepths)
                    depth_bound = depth_bound.astype(np.float32)
                else:
                    depth_interval = (depth_max - depth_min) / self.ndepths
                    depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

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

        scaled_input_images, scaled_input_cams = scale_dtu_input(imgs, cams, scale=resize_scale)

        # crop to fit network
        croped_imgs, croped_cams = crop_dtu_input(scaled_input_images, scaled_input_cams,
                                                  height=self.max_h, width=self.max_w,
                                                  base_image_size=self.base_image_size)

        ref_image = croped_imgs[0].copy()
####################################################################################################################
        original_image = ref_image  # [H, W, 3]
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
                "extrinsics_list": extrinsics_list,
                "intrinsics_list": intrinsics_list,
                'image': image_hr,
                'hr_coord': hr_coord,
                "depth_params": depth_params,
                "ref_image": ref_image,
                "lr_image": lr_image,
                "depth_bound": depth_bound
                }
