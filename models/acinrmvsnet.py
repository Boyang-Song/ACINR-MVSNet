import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from .module import *
from .feature_fetcher import *
from .inr import *
import math

class FeaturePyNet(nn.Module):
    """Feature Extraction Network: to extract features of original images from each view"""

    def __init__(self):
        """Initialize different layers in the network"""

        super(FeaturePyNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.inner3 = nn.Conv2d(8, 64, 1, bias=True)
        self.output4 = ConvBnReLU(64, 8, kernel_size=1, stride=1, pad=0)

        # [B,16,H/2,W/2]
        self.conv11_0 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv11_1 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv11_2 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv12_0 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv12_1 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv12_2 = nn.Conv2d(32, 32, 1, bias=False)
    def forward(self, x: torch.Tensor):

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        del conv7
        del conv10

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        del conv4

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner3(conv1)
        del conv1

        output_feature = self.output4(intra_feat)
        del intra_feat

        output_feature = self.conv11_2(self.conv11_1(self.conv11_0(output_feature)))
        output_feature = self.conv12_2(self.conv12_1(self.conv12_0(output_feature)))

        return output_feature

class VisibilityNet(nn.Module):
    """VisibilityNet: """
    def __init__(self, G):
        """Initialize method"""
        super(VisibilityNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=4, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=4, out_channels=1, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor):
        """Forward method for VisibilityNet"""
        return self.output(self.conv2(self.conv1(self.conv0(x1))))


class VolumeConv(nn.Module):
    """3D Regularization Network: to regularize cost volume"""

    def __init__(self, in_channels, base_channels):
        super(VolumeConv, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.conv1_0 = ConvBnReLU3D(in_channels, base_channels * 2, 3, stride=2, pad=1)
        self.conv2_0 = ConvBnReLU3D(base_channels * 2, base_channels * 4, 3, stride=2, pad=1)
        self.conv3_0 = ConvBnReLU3D(base_channels * 4, base_channels * 8, 3, stride=2, pad=1)

        self.conv0_1 = ConvBnReLU3D(in_channels, base_channels, 3, 1, pad=1)

        self.conv1_1 = ConvBnReLU3D(base_channels * 2, base_channels * 2, 3, 1, pad=1)
        self.conv2_1 = ConvBnReLU3D(base_channels * 4, base_channels * 4, 3, 1, pad=1)
        self.conv3_1 = ConvBnReLU3D(base_channels * 8, base_channels * 8, 3, 1, pad=1)

        self.conv4_0 = DeConvBnReLU3D(base_channels * 8, base_channels * 4, 3, 2, pad=1, out_pad=1)
        self.conv5_0 = DeConvBnReLU3D(base_channels * 4, base_channels * 2, 3, 2, pad=1, out_pad=1)
        self.conv6_0 = DeConvBnReLU3D(base_channels * 2, base_channels, 3, 2, pad=1, out_pad=1)

        self.conv6_2 = nn.Conv3d(base_channels, 1, 3, padding=1, bias=False)

    def forward(self, x):
        conv0_1 = self.conv0_1(x)

        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)

        conv1_1 = self.conv1_1(conv1_0)
        conv2_1 = self.conv2_1(conv2_0)
        conv3_1 = self.conv3_1(conv3_0)

        conv4_0 = self.conv4_0(conv3_1)

        conv5_0 = self.conv5_0(conv4_0 + conv2_1)
        conv6_0 = self.conv6_0(conv5_0 + conv1_1)

        conv6_2 = self.conv6_2(conv6_0 + conv0_1)

        return conv6_2

class EGNPyNet(nn.Module):
    """to extract features for Enhanced GN"""

    def __init__(self, forTNT):
        """Initialize different layers in the network"""

        super(EGNPyNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.inner3 = nn.Conv2d(8, 64, 1, bias=True)
        self.output4 = ConvBnReLU(64, 8, kernel_size=1, stride=1, pad=0)

        if forTNT:
            # [B,16,H/2,W/2]
            self.conv11_0 = ConvBnReLU(8, 16, 5, 2, 2)
            self.conv11_1 = ConvBnReLU(16, 16, 3, 1, 1)
            self.conv11_2 = ConvBnReLU(16, 16, 3, 1, 1)
            # [B,32,H/2,W/2]
            self.conv12_0 = ConvBnReLU(16, 32, kernel_size=1, stride=1, pad=0)
            self.conv12_1 = ConvBnReLU(32, 32, 3, 1, 1)
            self.conv12_2 = nn.Conv2d(32, 32, 1, bias=False)
        else:
            # [B,16,H,W]
            self.conv11_0 = ConvBnReLU(8, 16, kernel_size=1, stride=1, pad=0)
            self.conv11_1 = ConvBnReLU(16, 16, 3, 1, 1)
            self.conv11_2 = ConvBnReLU(16, 16, 3, 1, 1)
            # [B,32,H,W]
            self.conv12_0 = ConvBnReLU(16, 32, kernel_size=1, stride=1, pad=0)
            self.conv12_1 = ConvBnReLU(32, 32, 3, 1, 1)
            self.conv12_2 = nn.Conv2d(32, 32, 1, bias=False)
    def forward(self, x: torch.Tensor):

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        intra_feat = F.interpolate(conv10, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner1(conv7)
        del conv7
        del conv10

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        del conv4

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner3(conv1)
        del conv1

        output_feature = self.output4(intra_feat)
        del intra_feat

        output_feature = self.conv11_2(self.conv11_1(self.conv11_0(output_feature)))
        output_feature = self.conv12_2(self.conv12_1(self.conv12_0(output_feature)))

        return output_feature

def disp_to_depth(disp, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    scaled_disp = scaled_disp.clamp(min = 1e-4)
    depth = 1 / scaled_disp
    return depth

def depth_to_disp(depth, min_depth, max_depth):
    scaled_disp = 1 / depth
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = (scaled_disp - min_disp) / ((max_disp - min_disp))
    return disp

class ACINRMVSNet(nn.Module):
    def __init__(self, max_h, max_w, forTNT=False, isTest=False):
        super(ACINRMVSNet, self).__init__()
        self.forTNT = forTNT
        self.isTest = isTest
        self.dinp_h = max_h//4
        self.dinp_w = max_w//4
        self.G = 4


        self.feature_fetcher = FeatureFetcher()
        self.feature_grad_fetcher = FeatureGradFetcher()
        self.point_grad_fetcher = PointGrad()

        self.coarse_img_conv = FeaturePyNet()
        self.visibilitynet = VisibilityNet(self.G)
        self.coarse_vol_conv = VolumeConv(self.G, 8)
        self.jiif_net = INR()
        self.flow_img_conv = EGNPyNet(self.forTNT)

    def forward(self, imgs, proj_matrices, depth_values, extrinsics_list, intrinsics_list,\
                image, hr_coord, lr_image, depth_bound):
        preds = collections.OrderedDict()

        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        batch_size, num_depth = depth_values.shape[0], depth_values.shape[1]
        num_views = len(imgs)
#################### Coarse MVSNet ####################
        # step 1. feature extraction
        features = [self.coarse_img_conv(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        B,C,H,W=ref_feature.shape

        ################ ACINR V3 ##################
        # step 2. differentiable homograph, build cost volume
        ref_feature = ref_feature.view(B, self.G, C // self.G, 1, H, W)
        similarity_sum = 0.0
        pixel_wise_weight_sum = 0.0
        for src_feature, src_proj in zip(src_features, src_projs):
            warped_feature = homo_warping(src_feature, src_proj, ref_proj, depth_values)
            warped_feature = warped_feature.view(B, self.G, C // self.G, num_depth, H, W)
            # group-wise correlation
            similarity = (warped_feature * ref_feature).mean(2)
            reweight = self.visibilitynet(similarity)  # B, 1, D, H, W

            if self.isTest:
                similarity_sum += reweight * similarity
                pixel_wise_weight_sum += reweight
            else:
                similarity_sum = similarity_sum + reweight * similarity
                pixel_wise_weight_sum = pixel_wise_weight_sum + reweight

            del warped_feature, similarity, reweight
        # aggregated matching cost across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum)  # [B, G, Ndepth, H, W]
        del similarity_sum, ref_feature

        # step 3. cost volume regularization
        cost_reg = self.coarse_vol_conv(similarity)
        cost_reg = cost_reg.squeeze(1)

        prob_volume = F.softmax(-cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)
        depth = depth.unsqueeze(1)

        if self.isTest:
            with torch.no_grad():
                # photometric confidence
                prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
                depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
                photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1))

            if self.forTNT:
                photometric_confidence = F.interpolate(photometric_confidence, (self.dinp_h * 2, self.dinp_w * 2), \
                                                       mode="bilinear", align_corners=False)
            else:
                photometric_confidence = F.interpolate(photometric_confidence, (self.dinp_h * 4, self.dinp_w * 4), \
                                                       mode="bilinear", align_corners=False)

            preds["photometric_confidence"] = photometric_confidence
#################### Implict Neural SR/Propagation ####################
        preds["coarse_depth_map"] = depth
        depth_min_ = depth_bound[:, 0, None, None, None]
        depth_max_ = depth_bound[:, -1, None, None, None]

        inv_cur_depth = depth_to_disp(depth, depth_min_, depth_max_)
        inv_depth = self.jiif_net(inv_cur_depth, image, hr_coord, self.dinp_h * 2, self.dinp_w * 2, lr_image)
        inv_depth = inv_depth.reshape(batch_size, 1, self.dinp_h * 2, self.dinp_w * 2)# (B, 1, FH, FW)
        depth = disp_to_depth(inv_depth, depth_min_, depth_max_)

        preds["flow1"] = depth
#################### Gauss-Newton Refinement ####################
        def gn_update(estimated_depth_map, image_scale):
            flow_height = int(img_height * image_scale)
            flow_width = int(img_width * image_scale)
            estimated_depth_map = F.interpolate(estimated_depth_map, (flow_height, flow_width), mode="bilinear", align_corners=False)

            if self.isTest:
                estimated_depth_map = estimated_depth_map.detach()

            # GN step
            cam_extrinsic = extrinsics_list[:, :, :3, :4].clone()
            R = cam_extrinsic[:, :, :3, :3]
            t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
            R_inv = torch.inverse(R)
            cam_intrinsic = intrinsics_list.clone()
            if self.isTest:
                cam_intrinsic[:, :, :2, :3] *= image_scale
            else:
                cam_intrinsic[:, :, :2, :3] *= (4 * image_scale)

            ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
            feature_map_indices_grid = get_pixel_grids(flow_height, flow_width) \
                .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(imgs[0].device)

            uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                              feature_map_indices_grid)  # (B, 1, 3, FH*FW)

            interval_depth_map = estimated_depth_map
            cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
            world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

            grad_pts = self.point_grad_fetcher(world_points, cam_intrinsic, cam_extrinsic)

            R_tar_ref = torch.bmm(R.view(batch_size * num_views, 3, 3),
                                  R_inv[:, 0:1, :, :].repeat(1, num_views, 1, 1).view(batch_size * num_views, 3, 3))

            R_tar_ref = R_tar_ref.view(batch_size, num_views, 3, 3)
            d_pts_d_d = uv.unsqueeze(-1).permute(0, 1, 3, 2, 4).contiguous().repeat(1, num_views, 1, 1, 1)
            d_pts_d_d = R_tar_ref.unsqueeze(2) @ d_pts_d_d
            d_uv_d_d = torch.bmm(grad_pts.view(-1, 2, 3), d_pts_d_d.view(-1, 3, 1)).view(batch_size, num_views, 1,
                                                                                         -1, 2, 1)

            # !!! need to redefine EGNPyNet for different image_scale
            all_features = [self.flow_img_conv(img) for img in imgs]
            all_features = torch.stack(all_features, dim=1)
            # print(all_features.shape)

            if self.isTest:
                point_features, point_features_grad = \
                    self.feature_grad_fetcher.test_forward(all_features, world_points, cam_intrinsic, cam_extrinsic)
            else:
                point_features, point_features_grad = \
                    self.feature_grad_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

            c = all_features.size(2)
            d_uv_d_d_tmp = d_uv_d_d.repeat(1, 1, c, 1, 1, 1)
            # print("d_uv_d_d tmp size:", d_uv_d_d_tmp.size())
            J = point_features_grad.view(-1, 1, 2) @ d_uv_d_d_tmp.view(-1, 2, 1)
            J = J.view(batch_size, num_views, c, -1, 1)[:, 1:, ...].contiguous()\
                .permute(0, 3, 1, 2, 4).contiguous().view(-1, c * (num_views - 1), 1)

            # print(J.size())
            resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
            first_resid = torch.sum(torch.abs(resid), dim=(1, 2))
            # print(resid.size())
            resid = resid.permute(0, 3, 1, 2).contiguous().view(-1, c * (num_views - 1), 1)

            J_t = torch.transpose(J, 1, 2)
            H = J_t @ J
            b = -J_t @ resid
            delta = b / (H + 1e-6)
            # #print(delta.size())
            _, _, h, w = estimated_depth_map.size()
            flow_result = estimated_depth_map + delta.view(-1, 1, h, w)

            # check update results
            interval_depth_map = flow_result
            cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
            world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

            point_features = \
                self.feature_fetcher(all_features, world_points, cam_intrinsic, cam_extrinsic)

            resid = point_features[:, 1:, ...] - point_features[:, 0:1, ...]
            second_resid = torch.sum(torch.abs(resid), dim=(1, 2))
            # print(first_resid.size(), second_resid.size())

            # only accept good update
            flow_result = torch.where((second_resid < first_resid).view(batch_size, 1, flow_height, flow_width),
                                      flow_result, estimated_depth_map)
            return flow_result


        if self.isTest:
            depth = torch.detach(depth)
        if self.forTNT:
            flow = gn_update(depth, image_scale=0.5)
        else:
            flow = gn_update(depth, image_scale=1)

        preds["flow2"] = flow
#################### Return ####################
        preds["depth"] = flow.squeeze(1)
        if self.isTest:
            return {"coarse_depth": preds["coarse_depth_map"], "flow1": preds["flow1"], "flow2": preds["flow2"],\
                    "depth": preds["depth"], "photometric_confidence": preds["photometric_confidence"]}
        else:
            return {"coarse_depth": preds["coarse_depth_map"], "flow1": preds["flow1"], "flow2": preds["flow2"],\
                "depth": preds["depth"], "photometric_confidence": torch.empty(0)}


class MAELoss(nn.Module):
    def forward(self, pred_depth_image, gt_depth_image, depth_interval):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        depth_interval = depth_interval.view(-1)
        mask_valid = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae

class MVS_MAE_Loss(nn.Module):
    def __init__(self):
        super(MVS_MAE_Loss, self).__init__()
        self.maeloss = MAELoss()

    def forward(self, coarse_depth_map, flow1, flow2, gt_depth_img, depth_interval):
        # print("PointMVS_MAE_Loss: depth_interval", depth_interval)
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        coarse_loss = self.maeloss(coarse_depth_map, resize_gt_depth, depth_interval)

        losses = {}
        losses["coarse_loss"] = coarse_loss

        resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
        flow1_loss = self.maeloss(flow1, resize_gt_depth, depth_interval)
        losses["flow1_loss"] = flow1_loss

        resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
        flow2_loss = self.maeloss(flow2, resize_gt_depth, depth_interval)
        losses["flow2_loss"] = flow2_loss

        return losses

def cal_less_percentage(pred_depth, gt_depth, depth_interval, threshold):
    shape = list(pred_depth.size())
    mask_valid = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    denom = torch.sum(mask_valid) + 1e-7
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct

class BlendMetric(nn.Module):
    def __init__(self):
        super(BlendMetric, self).__init__()

    def forward(self, coarse_depth_map, flow1, flow2, gt_depth_img, depth_interval):

        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        less_one_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 3.0)

        metrics = {
            "<1_pct_cor": less_one_pct_coarse,
            "<3_pct_cor": less_three_pct_coarse,
        }

        resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
        less_one_pct_flow1 = cal_less_percentage(flow1, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_flow1 = cal_less_percentage(flow1, resize_gt_depth, depth_interval, 3.0)

        metrics["<1_pct_flow1"] = less_one_pct_flow1
        metrics["<3_pct_flow1"] = less_three_pct_flow1

        resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
        less_one_pct_flow2 = cal_less_percentage(flow2, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_flow2 = cal_less_percentage(flow2, resize_gt_depth, depth_interval, 3.0)

        metrics["<1_pct_flow2"] = less_one_pct_flow2
        metrics["<3_pct_flow2"] = less_three_pct_flow2

        return metrics