import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .module import *

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    ranged in [-1, 1]
    e.g.
        shape = [2] get (-0.5, 0.5)
        shape = [3] get (-0.67, 0, 0.67)
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) # [H, W, 2]
    if flatten:
        ret = ret.view(-1, ret.shape[-1]) # [H*W, 2]
    return ret

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True) # [H*W, 2]
    pixel = depth.view(-1, 1) # [H*W, 1]
    return coord, pixel

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class ImageFeatPyNet(nn.Module):
    """to extract Image LATENT CODE for INR """

    def __init__(self):
        """Initialize different layers in the network"""

        super(ImageFeatPyNet, self).__init__()

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

        self.inner2 = nn.Conv2d(16, 32, 1, bias=True)
        self.inner3 = nn.Conv2d(8, 32, 1, bias=True)

        self.output = nn.Conv2d(32, 32, 1, bias=False)

    def forward(self, x: torch.Tensor):

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))

        intra_feat = F.interpolate(
            conv7, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        del conv7
        del conv4

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner3(conv1)
        del conv1

        output_feature = self.output(intra_feat)
        del intra_feat

        return output_feature

class DepthFeatPyNet(nn.Module):
    """to extract DEPTH LATENT CODE for INR """

    def __init__(self):
        """Initialize different layers in the network"""

        super(DepthFeatPyNet, self).__init__()

        self.conv0 = ConvBnReLU(1, 8, 3, 1, 1)
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

        self.inner2 = nn.Conv2d(16, 32, 1, bias=True)
        self.inner3 = nn.Conv2d(8, 32, 1, bias=True)

        self.output = nn.Conv2d(32, 32, 1, bias=False)

    def forward(self, x: torch.Tensor):

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))

        intra_feat = F.interpolate(
            conv7, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner2(conv4)
        del conv7
        del conv4

        intra_feat = F.interpolate(
            intra_feat, scale_factor=2.0, mode="bilinear", align_corners=False) + self.inner3(conv1)
        del conv1

        output_feature = self.output(intra_feat)
        del intra_feat

        return output_feature

class INR(nn.Module):

    def __init__(self,mlp_dim=[256,128,64,32]):
        super().__init__()
        self.mlp_dim = mlp_dim

        self.image_encoder = ImageFeatPyNet()
        self.depth_encoder = DepthFeatPyNet()

        imnet_in_dim = 32 + 32 * 2 + 2

        self.imnet = MLP(imnet_in_dim, out_dim=2, hidden_list=self.mlp_dim)

    def query(self, feat, coord, hr_guide, lr_guide):

        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W

        b, c, h, w = feat.shape  # lr
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', \
                                   padding_mode='border',align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []

        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx
                coord_[:, :, 1] += (vy) * ry
                k += 1

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest',\
                                       padding_mode='border',align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', \
                                        padding_mode='border',align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest',\
                                           padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [B, N, C]
                q_guide = torch.cat([q_guide_lr, q_guide_hr - q_guide_lr], dim=-1)

                inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1)  # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # [B, N, 2, kk]
        weight = F.softmax(preds[:, :, 1, :], dim=-1)

        ret = (preds[:, :, 0, :] * weight).sum(-1, keepdim=True)

        return ret

    def forward(self, depth, image, coord, H, W, lr_image):

        feat = self.depth_encoder(depth)
        depth_lr_up = F.interpolate(depth, (H, W), mode="bicubic", align_corners=False)
        res = depth_lr_up.reshape(depth.shape[0], H*W, 1)

        hr_guide = self.image_encoder(image)
        lr_guide = self.image_encoder(lr_image)

        N = coord.shape[1]  # coord ~ [B, N, 2]
        n = 30720

        if N<=n:
            res = res + self.query(feat, coord, hr_guide, lr_guide)
        else:
            tmp = []
            for start in range(0, N, n):
                end = min(N, start + n)
                ans = self.query(feat, coord[:, start:end], hr_guide, lr_guide) # [B, N, 1]
                tmp.append(ans)
            res = res + torch.cat(tmp, dim=1)

        return res

