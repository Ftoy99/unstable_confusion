import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout=0.):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.temb_proj = nn.Linear(temb_dim, out_ch)
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = torch.silu(h)
        h = h + self.temb_proj(torch.silu(temb))[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = torch.silu(h)
        h = self.dropout(h)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, temb_dim):
        super(AttentionBlock, self).__init__()
        self.q = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.k = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.v = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        # Projection of timestep embedding to match the attention projection size
        self.temb_proj = nn.Linear(temb_dim, in_ch)

    def forward(self, x, temb):
        # Apply timestep embedding to queries, keys, and values
        q = self.q(x) + self.temb_proj(temb)[:, :, None, None]  # Add timestep embedding to query
        k = self.k(x) + self.temb_proj(temb)[:, :, None, None]  # Add timestep embedding to key
        v = self.v(x)  # No timestep embedding added to values, typically

        # Attention weights calculation
        w = torch.einsum('bchw,bCHW->bhwHW', q, k) * (q.shape[1] ** -0.5)
        w = w.view(w.size(0), w.size(2), w.size(3), -1)
        w = torch.softmax(w, dim=-1)
        w = w.view(w.size(0), w.size(2), w.size(3), w.size(4), w.size(5))

        # Apply attention weights to the values and project the output
        h = torch.einsum('bhwHW,bHWc->bhwc', w, v)
        return x + self.proj_out(h)


class UNetModel(nn.Module):
    def __init__(self, image_size=64, in_channels=3, out_channels=3, model_channels=224,
                 attention_resolutions=[8, 4, 2], num_res_blocks=2, channel_mult=[1, 2, 3, 4],
                 num_head_channels=32, dropout=0.):
        super(UNetModel, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_head_channels = num_head_channels
        self.dropout = dropout

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, stride=1, padding=1)

        # Embedding for timestep (temb)
        self.temb_proj = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )

        # ResNet blocks
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Downsampling and ResNet blocks
        for i, mult in enumerate(channel_mult):
            down_block = []
            for _ in range(num_res_blocks):
                down_block.append(ResNetBlock(model_channels * channel_mult[i],
                                              model_channels * mult, model_channels, dropout))
            self.down_blocks.append(nn.ModuleList(down_block))

        # Attention blocks
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(model_channels * mult, model_channels) for mult in channel_mult
        ])

        # Final convolution output layer
        self.conv_out = nn.Conv2d(model_channels * channel_mult[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, temb):
        B, C, H, W = x.shape

        # Initial convolution
        h = self.conv_in(x)

        hs = [h]

        # Downsampling and residual blocks
        for i, down_block in enumerate(self.down_blocks):
            for block in down_block:
                h = block(h, temb)
            if H in self.attention_resolutions:
                h = self.attn_blocks[i](h, temb)
            hs.append(h)
            if i != len(self.down_blocks) - 1:
                h = torch.avg_pool2d(h, 2, stride=2)

        # Middle
        h = self.down_blocks[-1][-1](h, temb)
        if H in self.attention_resolutions:
            h = self.attn_blocks[-1](h, temb)

        # Up sampling
        for i in reversed(range(len(self.down_blocks))):
            for block in self.down_blocks[i]:
                h = block(h, temb)
            if H in self.attention_resolutions:
                h = self.attn_blocks[i](h, temb)
            if i != 0:
                h = torch.nn.functional.interpolate(h, scale_factor=2, mode='nearest')

        # Final output layer
        h = torch.silu(h)
        h = self.conv_out(h)

        return h