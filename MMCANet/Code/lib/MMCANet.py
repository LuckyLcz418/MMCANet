import torch
from torch import nn
import torch.nn.functional as F
from Code.lib.SSA import shunted_s
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from Code.lib.DCAFM import CSBlock

from Code.lib.CSAM import CPA_43, CPA_432, CPA_4321

# from models.CSFM import CPA_concate_43, CPA_concate_432, CPA_concate_4321
from einops import rearrange

# from featureVisualization import vis_feat


def conv_bn_relu_1133(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1, 0),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


# 模态特征concate融合并降维至128
class feature_fuse(nn.Module):
    def __init__(self, in_channel=128, out_channel=128):
        super(feature_fuse, self).__init__()
        self.dim = in_channel
        self.out_dim = out_channel
        self.fuseconv = nn.Sequential(
            nn.Conv2d(2 * self.dim, self.out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.out_dim),
            nn.ReLU(True),
        )
        self.norm_rgb = nn.BatchNorm2d(self.dim)
        self.norm_d = nn.BatchNorm2d(self.dim)

    def forward(self, Ri, Di):
        assert Ri.ndim == 4  # 检查Ri是否是一个四维张量

        Ri = self.norm_rgb(Ri)
        Di = self.norm_d(Di)

        RDi = torch.cat((Ri, Di), dim=1)  # 128*2
        RDi = self.fuseconv(RDi)  # 128，降维
        RDi = self.conv(RDi)  # 3*3卷积
        return RDi


class Decoder(nn.Module):
    def __init__(self, dim=128):
        super(Decoder, self).__init__()

        self.dim = dim
        self.out_dim = dim

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear")

        self.Conv43 = conv_bn_relu_1133(256, 128)
        self.Conv432 = conv_bn_relu_1133(256, 128)
        self.Conv4321 = conv_bn_relu_1133(256, 128)

        # 通道注意力 + 空间注意力
        self.modality_fuse1 = CSBlock(
            in_dim=64, key_dim=64, value_dim=64, head_count=2, token_mlp="mix_skip"
        )
        self.modality_fuse2 = CSBlock(
            in_dim=128, key_dim=128, value_dim=128, head_count=4, token_mlp="mix_skip"
        )
        self.modality_fuse3 = CSBlock(
            in_dim=256, key_dim=256, value_dim=256, head_count=8, token_mlp="mix_skip"
        )
        self.modality_fuse4 = CSBlock(
            in_dim=512, key_dim=512, value_dim=512, head_count=16, token_mlp="mix_skip"
        )

        self.cpa_43 = CPA_43(dim=128, heads=2, mlp_dim=512, dropout=0.0)
        self.cpa_432 = CPA_432(dim=128, heads=2, mlp_dim=512, dropout=0.0)
        self.cpa_4321 = CPA_4321(dim=128, heads=2, mlp_dim=512, dropout=0.0)

        self.fuse1 = feature_fuse(in_channel=64, out_channel=128)
        self.fuse2 = feature_fuse(in_channel=128, out_channel=128)
        self.fuse3 = feature_fuse(in_channel=256, out_channel=128)
        self.fuse4 = feature_fuse(in_channel=512, out_channel=128)

       
        self.sal_pred = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )

    def forward(self, x, feature_list, feature_list_depth):
        # 特征提取
        R1, R2, R3, R4 = (
            feature_list[0],
            feature_list[1],
            feature_list[2],
            feature_list[3],
        )
        D1, D2, D3, D4 = (
            feature_list_depth[0],
            feature_list_depth[1],
            feature_list_depth[2],
            feature_list_depth[3],
        )

        R1, D1 = self.modality_fuse1(R1, D1)
        R2, D2 = self.modality_fuse2(R2, D2)
        R3, D3 = self.modality_fuse3(R3, D3)
        R4, D4 = self.modality_fuse4(R4, D4)
        

        fuse1 = self.fuse1(R1, D1)
        fuse2 = self.fuse2(R2, D2)
        fuse3 = self.fuse3(R3, D3)
        fuse4 = self.fuse4(R4, D4)
        
        x4 = fuse4.flatten(2).transpose(1, 2)

        B3, C3, H3, W3 = fuse3.shape
        x3 = fuse3.flatten(2).transpose(1, 2)

        new_fuse3 = self.cpa_43(x3, x4)

        B2, C2, H2, W2 = fuse2.shape
        x2 = fuse2.flatten(2).transpose(1, 2)
        new_fuse2 = self.cpa_432(x2, new_fuse3, x4)

        B1, C1, H1, W1 = fuse1.shape
        x1 = fuse1.flatten(2).transpose(1, 2)
        new_fuse1 = self.cpa_4321(x1, new_fuse2, new_fuse3, x4)

        new_fuse3 = new_fuse3.permute(0, 2, 1).reshape(B3, C3, H3, W3)
        new_fuse2 = new_fuse2.permute(0, 2, 1).reshape(B2, C2, H2, W2)
        new_fuse1 = new_fuse1.permute(0, 2, 1).reshape(B1, C1, H1, W1)


        # 解码
        RD43 = self.up2(fuse4)
        RD43 = torch.cat((RD43, new_fuse3), dim=1)
        RD43 = self.Conv43(RD43)

        RD432 = self.up2(RD43)
        RD432 = torch.cat((RD432, new_fuse2), dim=1)
        RD432 = self.Conv432(RD432)

        RD4321 = self.up2(RD432)
        RD4321 = torch.cat((RD4321, new_fuse1), dim=1)
        RD4321 = self.Conv4321(RD4321)  # [B, 128, 56, 56]

        sal_map = self.up4(RD4321)
        sal_map = self.sal_pred(sal_map)

        return sal_map


class MMCANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_rgb = shunted_s(pretrained=True)
        self.encoder_depth = shunted_s(pretrained=True)
        self.decoder = Decoder()

    def forward(self, input_rgb, input_depth):
        rgb_feats = self.encoder_rgb(input_rgb)
        depth_feats = self.encoder_depth(input_depth)

        sal_map = self.decoder(input_rgb, rgb_feats, depth_feats)

        return sal_map
