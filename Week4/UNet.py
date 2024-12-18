import math
from typing import Optional

import torch
from torch import nn, Tensor


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        super(TimestepEmbedding, self).__init__()
        self.linear1 = nn.Linear(n_channels // 4, n_channels)
        self.swish = Swish()
        self.linear2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.linear1(emb)
        emb = self.swish(emb)
        emb = self.linear2(emb)
        return emb


class DownSample(nn.Module):

    def __init__(self, n_channels):
        super(DownSample, self).__init__()
        # Kernel Strider Padding
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x, t):
        return self.conv(x)


class UpSample(nn.Module):

    def __init__(self, n_channels):
        super(UpSample, self).__init__()
        # Kernel Strider Padding
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x, t):
        return self.conv(x)


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_channels, has_attn, norm_group):
        super(DownBlock, self).__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels, norm_group)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=8)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):

    def __init__(self, n_channels, time_channels, norm_group):
        super(MiddleBlock, self).__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels, norm_group)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels, norm_group)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        return x


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super(UpBlock, self).__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=8)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, n_groups: int = 32, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.swish1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        self.time_swish = Swish()
        self.time_emb = nn.Linear(time_channels, out_channels)

        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.swish2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, t: Tensor):

        h = self.norm1(x)
        h = self.swish1(h)
        h = self.conv1(h)

        t = self.time_emb(t)

        h = h + t[:, :, None, None]

        h = self.norm2(h)
        h = self.swish2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = h + self.shortcut(x)

        return h


class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32, dropout: float = 0.1):
        super(AttentionBlock, self).__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape

        # Normalize
        x = self.norm(x)
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)  # [B, HW, C]

        # Q, K, V projection
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split into Q, K, V

        # Scaled Dot-Product Attention
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)

        # Reshape and output projection
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res = self.dropout(res)
        res += x  # Residual connection

        # Reshape back to [B, C, H, W]
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class UNet(nn.Module):

    def __init__(self, image_channels: int = 3, n_channels: int = 64, ch_mults=(1, 2, 2, 4),
                 is_attn=(False, False, True, True), norm_group=8, num_res_blocks=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.time_emb = TimestepEmbedding(n_channels * 4)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        self.down = nn.ModuleList()

        self.up = nn.ModuleList()

        n_resolutions = len(ch_mults)

        out_channels = in_channels = n_channels  # n channels are the channels of the image_proj
        # Down blocks
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(num_res_blocks):
                print(f"DownBlock {in_channels}->{out_channels}")
                down_block = DownBlock(in_channels, out_channels, time_channels=n_channels * 4, has_attn=is_attn[i],
                                       norm_group=norm_group)
                self.down.append(down_block)
                in_channels = out_channels
            if i < n_resolutions - 1:
                print(f"DownSample {out_channels}->{out_channels}")
                down_sample = DownSample(out_channels)
                self.down.append(down_sample)
                in_channels = out_channels

        self.middle = MiddleBlock(out_channels, n_channels * 4, norm_group)

        # Up Blocks
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            for _ in range(num_res_blocks):
                print(f"UpBlock {in_channels}->{out_channels}")
                up_block = UpBlock(in_channels, out_channels, time_channels=n_channels * 4, has_attn=is_attn[i])
                self.up.append(up_block)
            out_channels = in_channels // ch_mults[i]
            print(f"UpBlock {in_channels}->{out_channels}")
            up_block = UpBlock(in_channels, out_channels, time_channels=n_channels * 4, has_attn=is_attn[i])
            self.up.append(up_block)
            in_channels = out_channels
            if i > 0:
                print(f"UpSample {in_channels}->{in_channels}")
                up_sample = UpSample(in_channels)
                self.up.append(up_sample)

        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: Tensor, t: Tensor):
        B, C, H, W = x.shape

        t = self.time_emb(t)  # create embedding for timesteps

        x = self.image_proj(x)  # project the image to higher channels
        assert x.shape == (B, self.n_channels, H, W)

        skip = [x]

        # Down Blocks
        for i, block in enumerate(self.down):
            x = block(x, t)
            skip.append(x)

        x = self.middle(x, t)

        # Up Blocks
        for i, block in enumerate(self.up):
            if isinstance(block, UpSample):
                x = block(x, t)
            else:
                s = skip.pop()
                x = torch.cat((x, s), dim=1)
                x = block(x, t)

        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)
        return x
