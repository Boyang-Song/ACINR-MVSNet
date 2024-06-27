import argparse
import os
import numpy as np
from utils import *
import sys
from datasets.data_io import read_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

parser = argparse.ArgumentParser(
    description='Predict depth, filter, and fuse. May be different from the original implementation')

parser.add_argument('--testpath', default='/data/yhw/tanksAndtemple/', help='testing data path')
parser.add_argument('--testlist', default='./lists/tp_list.txt', help='testing scan list')

parser.add_argument('--outdir',
                    default='/data/yhw/github_pytorch/Dense-Mutli-Hypothesis-RMVS/outputs_tnt_20210426/0325_drmvs_g4_b1_d128_is1_mvsnet_cls_loss_ori_lr0.001_opAdam_ep10__shcosinedecay_FeatNet_AA_new_UNetConvLSTM_cg81_vn7_is0.25_gnTrue_v1.28_model_000007.ckpt/',
                    help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')
parser.add_argument('--test_dataset', choices=['dtu','tnt','blend'], help='which dataset to evaluate')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)




# read intrinsics and extrinsics
def read_camera_parameters(filename, scale_factors, scan):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[3])
    if scan not in scale_factors:
        # use the first cam to determine scale factor
        scale_factors[scan] = 100 / depth_min

    depth_min *= scale_factors[scan]
    depth_max *= scale_factors[scan]
    extrinsics[:3, 3] *= scale_factors[scan]
    return intrinsics, extrinsics, depth_max, depth_min


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data




# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    #sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LANCZOS4)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected = np.where(K_xyz_reprojected == 0, 1e-5, K_xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    xy_reprojected = np.clip(xy_reprojected, -1e8, 1e8)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks=[]
    for i in range(2,11):
        mask = np.logical_and(dist < i/4, relative_depth_diff < i/1300)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, plyfilename, scale_factors, scan):

    # the pair file
    pair_file = os.path.join(scan_folder, "cams/pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    # for each reference view and the corresponding source views
    ct2 = -1

    for ref_view, src_views in pair_data:

        ct2 += 1

        # load the camera parameters
        # print("ref view:scale_factors{}/scan{}", scale_factors, scan)
        ref_intrinsics, ref_extrinsics, ref_depth_max, ref_depth_min = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)), scale_factors, scan)
        # load the reference image
        #ref_img = read_img(os.path.join(results_folder, '{:08d}.jpg'.format(ref_view)))
        ref_img = read_img(os.path.join(scan_folder, 'blended_images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est_0/{:0>8}.pfm'.format(ref_view)))[0]

        h,w = ref_depth_est.shape
        import cv2
        # ref_img = cv2.pyrDown(ref_img)

        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence_0/{:0>8}.pfm'.format(ref_view)))[0]

        confidence = cv2.resize(confidence, (int(w), int(h)))
        photo_mask = confidence > 0.91

        all_srcview_depth_ests = []

        # compute the geometric mask
        geo_mask_sum = 0
        geo_mask_sums = []
        ct = 0
        for src_view in src_views:
            ct = ct + 1
            # camera parameters of the source view
            # print("src view:scale_factors{}/scan{}", scale_factors, scan)
            src_intrinsics, src_extrinsics, src_depth_max, src_depth_min = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)), scale_factors, scan)
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est_0/{:0>8}.pfm'.format(src_view)))[0]

            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                        ref_extrinsics,
                                                                                        src_depth_est,
                                                                                        src_intrinsics, src_extrinsics)

            if (ct==1):
                for i in range(2,11):
                    geo_mask_sums.append(masks[i-2].astype(np.int32))
            else :
                for i in range(2,11):
                    geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)

            geo_mask_sum += geo_mask.astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)

        geo_mask = geo_mask_sum >= 10

        for i in range (2,11):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        maskdepth = np.logical_and(depth_est_averaged > ref_depth_min, depth_est_averaged < ref_depth_max)
        if (not isinstance(geo_mask, bool)):

            final_mask = np.logical_and(photo_mask, geo_mask)
            final_mask = np.logical_and(final_mask, maskdepth)
            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)

            save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                        photo_mask.mean(),
                                                                                        geo_mask.mean(),
                                                                                        final_mask.mean()))

            if args.display:
                cv2.imshow('ref_img', ref_img[:, :, ::-1])
                cv2.imshow('ref_depth', ref_depth_est / 800)
                cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
                cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
                cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
                cv2.waitKey(0)

            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            # valid_points = np.logical_and(final_mask, ~used_mask[ref_view])
            valid_points = final_mask
            print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[:, :, :][valid_points]
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]
    scale_factors = {}
    for scan in scans:
        scan_folder = os.path.join(args.testpath, scan)
        out_folder = os.path.join(args.outdir, scan)
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'), scale_factors, scan)