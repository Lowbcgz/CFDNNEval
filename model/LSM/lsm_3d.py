"""
@origin author: Haixu Wu
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math


################################################################
# Multiscale modules 3D
################################################################

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



################################################################
# Patchify and Neural Spectral Block
################################################################
class NeuralSpectralBlock3d(nn.Module):
    def __init__(self, width, num_basis, patch_size=[8, 8, 4], num_token=4):
        super(NeuralSpectralBlock3d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
                                                                  dtype=torch.float).cuda()
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv3d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv3d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W, T = x.shape
        L = H * W * T
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = self.encoder_attn(x).view(B, C * 2, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3).contiguous()
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H, W, T = x.shape
        L = H * W * T
        latent_token = latent_token.view(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2).contiguous()
        x_tmp = self.decoder_attn(x).view(B, C, -1).permute(0, 2, 1).contiguous() \
            .view(B, L, self.head, C // self.head).permute(0, 2, 1, 3).contiguous()
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W, T) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W, T = x.shape
        # patchify
        x = x.view(x.shape[0], x.shape[1],
                   x.shape[2] // self.patch_size[0], self.patch_size[0], x.shape[3] // self.patch_size[1], self.patch_size[1],
                   x.shape[4] // self.patch_size[2], self.patch_size[2]).contiguous() \
            .permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous() \
            .view(x.shape[0] * (x.shape[2] // self.patch_size[0]) * (x.shape[3] // self.patch_size[1]) * (
                x.shape[4] // self.patch_size[2]), x.shape[1], self.patch_size[0], self.patch_size[1], self.patch_size[2])
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = x.view(B, (H // self.patch_size[0]), (W // self.patch_size[1]), (T // self.patch_size[2]), C,
                   self.patch_size[0], self.patch_size[1], self.patch_size[2]).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous() \
            .view(B, C, H, W, T).contiguous()
        return x


class LSM_3d(nn.Module):
    def __init__(self, inputs_channel, outputs_channel=None, d_model=32, num_token=4, num_basis=12, initial_step=1, n_case_params = 5, patch_size="",padding="", bilinear=True):
        super(LSM_3d, self).__init__()
        in_channels = inputs_channel*initial_step+1+3+n_case_params  # +1 for the mask, +3 for the grids
        if outputs_channel is None:
            out_channels = inputs_channel
        else:
            out_channels = outputs_channel
        width = d_model
        patch_size = [int(x) for x in patch_size.split(',')]
        padding = [int(x) for x in padding.split(',')]
        # multiscale modules
        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(width * 8, width * 16 // factor)
        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)
        # Patchified Neural Spectral Blocks
        self.process1 = NeuralSpectralBlock3d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock3d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock3d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock3d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock3d(width * 16 // factor, num_basis, patch_size, num_token)
        # projectors
        self.padding = padding
        self.patch_size = patch_size
        self.fc0 = nn.Linear(in_channels, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, case_params, mask, grid):
        _, H, W, D, _ = x.shape
        assert (H+self.padding[2]) % 16 == 0 and (W+self.padding[1]) % 16 == 0 and  (D+self.padding[0]) % 16 == 0, \
                            f"padding ({self.padding[0]},{self.padding[1]},{self.padding[2]}) is not suitable for (H,W,D)=({H},{W},{D}), the shape after padding should be divisible by 16."
        assert (H+self.padding[2]) % self.patch_size[0] == 0 and (W+self.padding[1]) % self.patch_size[1] == 0 and (D+self.padding[0]) % self.patch_size[2] ==0
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, mask, grid, case_params), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        if not all(item == 0 for item in self.padding):
            x = F.pad(x, [0, self.padding[0], 0, self.padding[1], 0, self.padding[2]])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            x = x[..., :-self.padding[2], :-self.padding[1], :-self.padding[0]]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x * mask
        return x
    
    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        pred = self(x, case_params, mask, grid)
        
        if loss_fn is not None:
            ## defined your specific loss calculations here
            loss = loss_fn(pred, y)
            return loss, pred, info
        else:
            #TODO: default loss_fn
            pass

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)