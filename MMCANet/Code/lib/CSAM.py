

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inplace_abn import InPlaceABN, InPlaceABNSync
from torch.nn import Softmax

def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

def conv3x3_bn_relu(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1, 0),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)




class EfficientAttention(nn.Module):  # this is multiAttention
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(self.in_channels, self.key_channels)
        self.queries = nn.Linear(self.in_channels, self.key_channels)
        self.values = nn.Linear(self.in_channels, self.value_channels)
        self.reprojection = nn.Linear(self.value_channels, self.in_channels)

    def forward(self, input_, x_pos_embed):
        B, N, C = input_.size()
        assert C == self.in_channels, "C {} != inchannels {}".format(
            C, self.in_channels
        )
        assert (
            input_.shape[1:] == x_pos_embed.shape[1:]
        ), "x.shape {} != x_pos_embed.shape {}".format(input_.shape, x_pos_embed.shape)
        keys = self.keys(input_ + x_pos_embed).permute(
            0, 2, 1
        )  # .reshape((n, self.key_channels, h * w))
        queries = self.queries(input_ + x_pos_embed).permute(
            0, 2, 1
        )  # .reshape(n, self.key_channels, h * w)
        values = self.values(input_).permute(
            0, 2, 1
        )  # .reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(
                keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2
            )
            query = F.softmax(
                queries[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            value = values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = context.transpose(1, 2) @ query

            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1, 2)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value


class Multi_EfficientAttention_43(nn.Module):  # this is multiAttention
    def __init__(self, dim, head_count, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.head_count = head_count

        self.queries = nn.Linear(self.dim, self.dim)
        self.keys = nn.Linear(self.dim, self.dim)
        self.values = nn.Linear(self.dim, self.dim)
        self.reprojection = nn.Sequential(
            nn.Linear(self.dim, self.dim), nn.Dropout(dropout)
        )

    def forward(self, f3, f4, f3_pos_embed, f4_pos_embed):
        # print(f3_pos_embed.shape)

        f3_queries = self.queries(f3 + f3_pos_embed).permute(
            0, 2, 1
        )  # .reshape(n, self.key_channels, h * w)
        f4_keys = self.keys(f4 + f4_pos_embed).permute(
            0, 2, 1
        )  # .reshape((n, self.key_channels, h * w))
        f4_values = self.values(f4).permute(
            0, 2, 1
        )  # .reshape((n, self.value_channels, h * w))
        head_key_channels = self.dim // self.head_count
        head_value_channels = self.dim // self.head_count

        attended_values = []
        for i in range(self.head_count):
            f4_key = F.softmax(
                f4_keys[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )
            f3_query = F.softmax(
                f3_queries[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            f4_value = f4_values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            context = f4_key @ f4_value.transpose(1, 2)

            attended_value = context.transpose(1, 2) @ f3_query
            attended_values.append(attended_value)
        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values = aggregated_values.transpose(1, 2)

        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value


class Multi_EfficientAttention_432(nn.Module):  # this is multiAttention
    def __init__(self, dim, head_count, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.head_count = head_count

        self.to_q1 = nn.Linear(self.dim, self.dim)
        self.to_k1 = nn.Linear(self.dim, self.dim)
        self.to_v1 = nn.Linear(self.dim, self.dim)
        self.to_q2 = nn.Linear(self.dim, self.dim)
        self.to_k2 = nn.Linear(self.dim, self.dim)
        self.to_v2 = nn.Linear(self.dim, self.dim)

        self.reprojection = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim), nn.Dropout(dropout)
        )

    def forward(self, f2, f3, f4, f2_pos_embed, f3_pos_embed, f4_pos_embed):
        f2_queries_1 = self.to_q1(f2 + f2_pos_embed).permute(0, 2, 1)
        f2_queries_2 = self.to_q2(f2 + f2_pos_embed).permute(0, 2, 1)

        f3_keys = self.to_k1(f3 + f3_pos_embed).permute(0, 2, 1)
        f3_values = self.to_v1(f3).permute(0, 2, 1)

        f4_keys = self.to_k2(f4 + f4_pos_embed).permute(0, 2, 1)
        f4_values = self.to_v2(f4).permute(0, 2, 1)

        head_key_channels = self.dim // self.head_count
        head_value_channels = self.dim // self.head_count

        attended_values_32 = []
        attended_values_42 = []
        for i in range(self.head_count):
            f2_query_1 = F.softmax(
                f2_queries_1[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )
            f3_key = F.softmax(
                f3_keys[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            f3_value = f3_values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            f2_query_2 = F.softmax(
                f2_queries_2[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )
            f4_key = F.softmax(
                f4_keys[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            f4_value = f4_values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            context32 = f3_key @ f3_value.transpose(1, 2)
            attended_value_32 = context32.transpose(1, 2) @ f2_query_1

            context42 = f4_key @ f4_value.transpose(1, 2)
            attended_value_42 = context42.transpose(1, 2) @ f2_query_2

            attended_values_32.append(attended_value_32)
            attended_values_42.append(attended_value_42)

        aggregated_values_32 = torch.cat(attended_values_32, dim=1)
        aggregated_values_32 = aggregated_values_32.transpose(1, 2)

        aggregated_values_42 = torch.cat(attended_values_42, dim=1)
        aggregated_values_42 = aggregated_values_42.transpose(1, 2)

        out = torch.cat((aggregated_values_32, aggregated_values_42), dim=-1)

        out = self.reprojection(out)

        return out


class Multi_EfficientAttention_4321(nn.Module):  # this is multiAttention
    def __init__(self, dim, head_count, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.head_count = head_count

        self.to_q1 = nn.Linear(self.dim, self.dim)
        self.to_k1 = nn.Linear(self.dim, self.dim)
        self.to_v1 = nn.Linear(self.dim, self.dim)
        self.to_q2 = nn.Linear(self.dim, self.dim)
        self.to_k2 = nn.Linear(self.dim, self.dim)
        self.to_v2 = nn.Linear(self.dim, self.dim)
        self.to_q3 = nn.Linear(self.dim, self.dim)
        self.to_k3 = nn.Linear(self.dim, self.dim)
        self.to_v3 = nn.Linear(self.dim, self.dim)

        self.reprojection = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim), nn.Dropout(dropout)
        )

    def forward(
        self, f1, f2, f3, f4, f1_pos_embed, f2_pos_embed, f3_pos_embed, f4_pos_embed
    ):
        f1_queries_1 = self.to_q1(f1 + f1_pos_embed).permute(0, 2, 1)
        f1_queries_2 = self.to_q2(f1 + f1_pos_embed).permute(0, 2, 1)
        f1_queries_3 = self.to_q3(f1 + f1_pos_embed).permute(0, 2, 1)

        f2_keys = self.to_k1(f2 + f2_pos_embed).permute(0, 2, 1)
        f2_values = self.to_v1(f2).permute(0, 2, 1)

        f3_keys = self.to_k2(f3 + f3_pos_embed).permute(0, 2, 1)
        f3_values = self.to_v2(f3).permute(0, 2, 1)

        f4_keys = self.to_k3(f4 + f4_pos_embed).permute(0, 2, 1)
        f4_values = self.to_v3(f4).permute(0, 2, 1)

        head_key_channels = self.dim // self.head_count
        head_value_channels = self.dim // self.head_count

        attended_values_21 = []
        attended_values_31 = []
        attended_values_41 = []
        for i in range(self.head_count):
            f1_query_1 = F.softmax(
                f1_queries_1[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )
            f2_key = F.softmax(
                f2_keys[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            f2_value = f2_values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            f1_query_2 = F.softmax(
                f1_queries_2[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )
            f3_key = F.softmax(
                f3_keys[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            f3_value = f3_values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            f1_query_3 = F.softmax(
                f1_queries_3[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )
            f4_key = F.softmax(
                f4_keys[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )
            f4_value = f4_values[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            context21 = f2_key @ f2_value.transpose(1, 2)
            attended_value_21 = context21.transpose(1, 2) @ f1_query_1

            context31 = f3_key @ f3_value.transpose(1, 2)
            attended_value_31 = context31.transpose(1, 2) @ f1_query_2

            context41 = f4_key @ f4_value.transpose(1, 2)
            attended_value_41 = context41.transpose(1, 2) @ f1_query_3

            attended_values_21.append(attended_value_21)
            attended_values_31.append(attended_value_31)
            attended_values_41.append(attended_value_41)

        aggregated_values_21 = torch.cat(attended_values_21, dim=1)
        aggregated_values_21 = aggregated_values_21.transpose(1, 2)

        aggregated_values_31 = torch.cat(attended_values_31, dim=1)
        aggregated_values_31 = aggregated_values_31.transpose(1, 2)

        aggregated_values_41 = torch.cat(attended_values_41, dim=1)
        aggregated_values_41 = aggregated_values_41.transpose(1, 2)

        out = torch.cat(
            (aggregated_values_21, aggregated_values_31, aggregated_values_41), dim=-1
        )

        out = self.reprojection(out)

        return out


class CPA_43(nn.Module):
    def __init__(self, dim=128, heads=2, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.norm3 = nn.LayerNorm(self.dim)

        self.f4_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=64, d_hid=self.dim),  # 64
            requires_grad=False,
        )
        self.f3_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=256, d_hid=self.dim),  # 256
            requires_grad=False,
        )

        self.Attention = Multi_EfficientAttention_43(
            dim, head_count=heads, dropout=dropout
        )
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, f3, f4):
        # attn + mlp
        f3 = f3 + self.drop_path(
            self.Attention(
                self.norm1(f3), self.norm2(f4), self.f3_pos_embed, self.f4_pos_embed
            )
        )
        f3 = f3 + self.drop_path(self.feedforward(self.norm3(f3)))
        return f3


def conv_bn_relu_1133(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, 1, 0),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class CPA_432(nn.Module):
    def __init__(self, dim=128, heads=2, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.f4_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=64, d_hid=self.dim),  # 64
            requires_grad=False,
        )
        self.f3_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=256, d_hid=self.dim),  # 256
            requires_grad=False,
        )
        self.f2_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=1024, d_hid=self.dim),  # 1024
            requires_grad=False,
        )

        self.Attention = Multi_EfficientAttention_432(
            dim, head_count=heads, dropout=dropout
        )
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, f2, f3, f4):
        f2 = f2 + self.drop_path(
            self.Attention(
                self.norm1(f2),
                self.norm2(f3),
                self.norm3(f4),
                self.f2_pos_embed,
                self.f3_pos_embed,
                self.f4_pos_embed,
            )
        )
        f2 = f2 + self.drop_path(self.feedforward(self.norm4(f2)))
        return f2


class CPA_4321(nn.Module):
    def __init__(self, dim=128, heads=2, mlp_dim=512, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)

        self.f4_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=64, d_hid=self.dim),
            requires_grad=False,
        )
        self.f3_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=256, d_hid=self.dim),
            requires_grad=False,
        )
        self.f2_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=1024, d_hid=self.dim),
            requires_grad=False,
        )
        self.f1_pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=4096, d_hid=self.dim),
            requires_grad=False,
        )

        self.Attention = Multi_EfficientAttention_4321(
            dim, head_count=heads, dropout=dropout
        )
        self.drop_path = nn.Dropout(dropout)
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, f1, f2, f3, f4):
        f1 = f1 + self.drop_path(
            self.Attention(
                self.norm1(f1),
                self.norm2(f2),
                self.norm3(f3),
                self.norm4(f4),
                self.f1_pos_embed,
                self.f2_pos_embed,
                self.f3_pos_embed,
                self.f4_pos_embed,
            )
        )
        f1 = f1 + self.drop_path(self.feedforward(self.norm5(f1)))
        return f1
