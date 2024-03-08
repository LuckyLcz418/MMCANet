import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SpatialCrossAttention(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys_rgb = nn.Conv2d(in_channels, key_channels, 1)
        self.queries_rgb = nn.Conv2d(in_channels, key_channels, 1)
        self.values_rgb = nn.Conv2d(in_channels, value_channels, 1)

        self.keys_T = nn.Conv2d(in_channels, key_channels, 1)
        self.queries_T = nn.Conv2d(in_channels, key_channels, 1)
        self.values_T = nn.Conv2d(in_channels, value_channels, 1)

        self.reprojection_rgb = nn.Conv2d(value_channels, in_channels, 1)
        self.reprojection_d = nn.Conv2d(value_channels, in_channels, 1)

        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_d = nn.Parameter(torch.zeros(1))

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, input_, input_T):
        n, c, h, w = input_.size()  # input: (N, C, H, W)
        # 对RGB特征做线性变换，生成Q, K, V
        keys_rgb = self.keys_rgb(input_).reshape(
            (n, self.key_channels, h * w)
        )  # (N, C, H*W)
        queries_rgb = self.queries_rgb(input_).reshape(
            n, self.key_channels, h * w
        )  # (N, C, H*W)
        values_rgb = self.values_rgb(input_).reshape(
            (n, self.value_channels, h * w)
        )  # (N, C, H*W)
        # 对T特征做线性变换，生成Q, K, V
        keys_T = self.keys_T(input_T).reshape(
            (n, self.key_channels, h * w)
        )  # (N, C, H*W)
        queries_T = self.queries_T(input_T).reshape(
            n, self.key_channels, h * w
        )  # (N, C, H*W)
        values_T = self.values_T(input_T).reshape(
            (n, self.value_channels, h * w)
        )  # (N, C, H*W)
        # 每一个头的通道数
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        attended_values_T = []
        for i in range(self.head_count):  # 根据注意力头划分通道，每个划分的注意力头的维度是32
            key_rgb = F.softmax(
                keys_rgb[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=2,
            )

            query_rgb = F.softmax(
                queries_rgb[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )

            value_rgb = values_rgb[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            key_T = F.softmax(
                keys_T[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2
            )

            query_T = F.softmax(
                queries_T[:, i * head_key_channels : (i + 1) * head_key_channels, :],
                dim=1,
            )

            value_T = values_T[
                :, i * head_value_channels : (i + 1) * head_value_channels, :
            ]

            context = key_rgb @ value_rgb.transpose(1, 2)  # dk*dv  (N, dC, dC)
            context_T = key_T @ value_T.transpose(1, 2)
            # context_fuse = (context + context_T) / 2

            attended_value_T = (context.transpose(1, 2) @ query_T).reshape(
                n, head_value_channels, h, w
            )  # dv * (h*w)
            attended_value = (context_T.transpose(1, 2) @ query_rgb).reshape(
                n, head_value_channels, h, w
            )
            attended_values.append(attended_value)
            attended_values_T.append(attended_value_T)
        # 将所有的头拼接起来
        aggregated_values = torch.cat(attended_values, dim=1)
        aggregated_values_T = torch.cat(attended_values_T, dim=1)

        reattn_rgb = self.reprojection_rgb(aggregated_values)
        reattn_T = self.reprojection_d(aggregated_values_T)

        # input_ = self.relu1(input_ + self.gamma_dr * attention)
        # input_T = self.relu2(input_T + self.gamma_rd * attention_T)

        return reattn_rgb, reattn_T


class ChannelCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_count=8,
        qkv_bias=False,
        qkv_T_bias=False,
        attn_drop=0,
        proj_drop=0,
    ):
        super().__init__()
        self.num_heads = head_count
        self.temperature_rgb = nn.Parameter(torch.ones(head_count, 1, 1))
        self.temperature_T = nn.Parameter(torch.ones(head_count, 1, 1))

        self.qkv_rgb = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_T = nn.Linear(dim, dim * 3, bias=qkv_T_bias)

        self.attn_drop_rgb = nn.Dropout(attn_drop)
        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_drop_rgb = nn.Dropout(proj_drop)

        self.attn_drop_T = nn.Dropout(attn_drop)
        self.proj_T = nn.Linear(dim, dim)
        self.proj_drop_T = nn.Dropout(proj_drop)

    def forward(self, x, x_d):
        B, N, C = x.shape

        qkv_rgb = self.qkv_rgb(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_rgb = qkv_rgb.permute(2, 0, 3, 1, 4)

        qkv_T = self.qkv_T(x_d).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_T = qkv_T.permute(2, 0, 3, 1, 4)

        q_rgb, k_rgb, v_rgb = (
            qkv_rgb[0],
            qkv_rgb[1],
            qkv_rgb[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        q_T, k_T, v_T = qkv_T[0], qkv_T[1], qkv_T[2]

        q_rgb = q_rgb.transpose(-2, -1)
        k_rgb = k_rgb.transpose(-2, -1)
        v_rgb = v_rgb.transpose(-2, -1)

        q_T = q_T.transpose(-2, -1)
        k_T = k_T.transpose(-2, -1)
        v_T = v_T.transpose(-2, -1)

        q_rgb = F.normalize(q_rgb, dim=-1)
        k_rgb = F.normalize(k_rgb, dim=-1)

        q_T = F.normalize(q_T, dim=-1)
        k_T = F.normalize(k_T, dim=-1)

        attn = (q_rgb @ k_T.transpose(-2, -1)) * self.temperature_rgb
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop_rgb(attn)
        x = (attn @ v_T).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_T = (q_T @ k_rgb.transpose(-2, -1)) * self.temperature_T
        attn_T = attn_T.softmax(dim=-1)
        attn_T = self.attn_drop_T(attn_T)
        x_d = (attn_T @ v_rgb).permute(0, 3, 1, 2).reshape(B, N, C)

        x = self.proj_rgb(x)
        x = self.proj_drop_rgb(x)

        x_d = self.proj_T(x_d)
        x_d = self.proj_drop_T(x_d)

        return x, x_d



class CSBlock(nn.Module):
    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.head_count = head_count
        # self.norm1_rgb = nn.BatchNorm2d(in_dim)
        # self.norm1_T = nn.BatchNorm2d(in_dim)
        self.norm1_rgb = nn.LayerNorm(in_dim)
        self.norm1_T = nn.LayerNorm(in_dim)

        self.attn = SpatialCrossAttention(
            in_channels=in_dim,
            key_channels=key_dim,
            value_channels=value_dim,
            head_count=self.head_count,
        )

        self.norm2_rgb = nn.LayerNorm(in_dim)
        self.norm2_T = nn.LayerNorm(in_dim)

        self.norm3_rgb = nn.LayerNorm(in_dim)
        self.norm3_T = nn.LayerNorm(in_dim)

        self.channel_attn = ChannelCrossAttention(in_dim)

        self.norm4_rgb = nn.LayerNorm(in_dim)
        self.norm4_T = nn.LayerNorm(in_dim)

        if token_mlp == "mix":
            self.mlp1_rgb = MixFFN(in_dim, int(in_dim * 4))
            self.mlp1_T = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2_rgb = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2_T = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1_rgb = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp1_T = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2_rgb = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2_T = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1_rgb = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp1_T = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2_rgb = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2_T = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x, x_d):
        # 转换输入的形状：(B, C, H, W) -> (B, N, C)
        B, D, H, W = x.shape
        x_reshape = x.flatten(2).transpose(1, 2)
        x_d_reshape = x_d.flatten(2).transpose(1, 2)

        # 输入特征做归一化，这里使用的输入形状为：(B, C, H, W)
        x_norm1 = self.norm1_rgb(x_reshape)
        x_d_norm1 = self.norm1_T(x_d_reshape)

        # 通道交叉注意力
        channel_attn_rgb, channel_attn_T = self.channel_attn(x_norm1, x_d_norm1)

        add1_rgb = x_reshape + channel_attn_rgb
        add1_T = x_d_reshape + channel_attn_T
        norm2_rgb = self.norm2_rgb(add1_rgb)
        norm2_T = self.norm2_T(add1_T)
        mlp1_rgb = self.mlp1_rgb(norm2_rgb, H, W)
        mlp1_T = self.mlp1_T(norm2_T, H, W)

        add2_rgb = add1_rgb + mlp1_rgb
        add2_T = add1_T + mlp1_T

        x_norm3 = self.norm3_rgb(add2_rgb)
        x_d_norm3 = self.norm3_T(add2_T)

        # 空间注意力
        x_norm3_new = x_norm3.permute(0, 2, 1).reshape(B, -1, H, W)
        x_d_norm3_new = x_d_norm3.permute(0, 2, 1).reshape(B, -1, H, W)

        spatial_attn_rgb, spatial_attn_T = self.attn(x_norm3_new, x_d_norm3_new)

        spatial_attn_rgb = spatial_attn_rgb.flatten(2).transpose(1, 2)
        spatial_attn_T = spatial_attn_T.flatten(2).transpose(1, 2)

        # FFN
        add3_rgb = add2_rgb + spatial_attn_rgb
        add3_T = add2_T + spatial_attn_T

        x_norm4 = self.norm4_rgb(add3_rgb)
        x_d_norm4 = self.norm4_T(add3_T)
        mlp2_rgb = self.mlp2_rgb(x_norm4, H, W)
        mlp2_T = self.mlp2_T(x_d_norm4, H, W)

        x_new = add3_rgb + mlp2_rgb
        x_d_new = add3_T + mlp2_T

        # 转换形状
        x_new = x_new.permute(0, 2, 1).reshape(B, -1, H, W)
        x_d_new = x_d_new.permute(0, 2, 1).reshape(B, -1, H, W)

        return x_new, x_d_new
        
