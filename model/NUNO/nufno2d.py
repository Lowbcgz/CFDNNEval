"""
Reference
----------
author:   Zongyi Li and Daniel Zhengyu Huang
source:   https://raw.githubusercontent.com/zongyi-li/Geo-FNO
reminder: slightly modified, e.g., file path, better output format, etc.
"""
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F



class NUNO_Data_utils_2d:
    def __init__(self) -> None:
        self.tree=None

    def _data_preprocessing(self, x,  grid, n_subdomains=8):

        """
        inputs:
            x: (T, n_x, n_c)
            grid: (n_x, dim)
        returns:
            input_u_sd_grid:  (T, s1_padded, s2_padded, n_subdomains, n_c)
            input_xy_sd:      (n_subdomains, maxlen_sd, 1, dim), point clouds in n_subdomains and are normalized to [-1,1], and padded to the maxLen.
            input_u_sd_mask:  (T, maxlen_sd, n_subdomains, 1）
            input_u_sd :      (T, maxlen_sd, n_subdomains, n_c)
        where input_xy_sd, input_u_sd_mask, input_u_sd are of point clouds format splited in `n` subdomins by KDTree,
            input_u_sd_grid is the interpolated result in `n` subdomains, resized to the same grid.
            
        Only generate KDTree for one time !!! This function will be first called in trainset. 
        """
        from .tree import KDTree
        from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
        with torch.no_grad():
            n_total = 1
            oversamp_ratio = 1.5
            input_xy = grid
            T, n_x, n_c = x.shape
            input_u = x.reshape(n_total, T, n_x, n_c).transpose((0, 2, 1, 3))  # (n_total, n_x, T, n_c)
            point_cloud = input_xy.tolist()  
            # Use kd-tree to generate subdomain division
            if self.tree == None:
                self.tree= KDTree(
                    point_cloud, dim=2, n_subdomains=n_subdomains, 
                    n_blocks=8, return_indices=True
                )
                self.tree.solve()
            # Gather subdomain info
            bbox_sd = self.tree.get_subdomain_bounding_boxes()
            indices_sd = self.tree.get_subdomain_indices()
            input_xy_sd = np.zeros((np.max([len(indices_sd[i]) for i in range(n_subdomains)]), n_subdomains, 2))  # (maxlen, n_subdomains, 2)
            for i in range(n_subdomains):
                # Normalize to [-1, 1]
                xy = input_xy[indices_sd[i], :]
                _min, _max = np.min(xy, axis=0, keepdims=True), \
                    np.max(xy, axis=0, keepdims=True)
                xy = (xy - _min) / (_max - _min) * 2 - 1
                # Long side alignment
                bbox = bbox_sd[i]
                if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                    xy = np.flip(xy, axis=1)
                input_xy_sd[:len(indices_sd[i]), i, :] = xy

            # print("Finish KD-Tree splitting")

            # t1 = default_timer()
            # print("Start interpolation...")
            # Calculate the padded grid size
            max_grid_size_x, max_grid_size_y = -1, -1
            grid_sizes = []
            is_transposed = [False] * n_subdomains
            for i in range(n_subdomains):
                n_points = len(indices_sd[i])
                bbox = bbox_sd[i]
                # Calculate the grid size, where the aspect ratio of the discrete grid 
                # remains the same as the that of the original subdomain (bbox)
                grid_size_x = np.sqrt(n_points * oversamp_ratio * \
                    (bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
                grid_size_y = grid_size_x * (bbox[1][1] - bbox[1][0]) / (bbox[0][1] - bbox[0][0])
                grid_size_x, grid_size_y = max(int(np.round(grid_size_x)), 2), \
                    max(int(np.round(grid_size_y)), 2)
                grid_sizes.append((grid_size_x, grid_size_y))
                # Long side alignment to reduce paddings
                if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                    grid_size_x, grid_size_y = grid_size_y, grid_size_x
                    is_transposed[i] = True
                max_grid_size_x, max_grid_size_y = max(max_grid_size_x, 
                    grid_size_x), max(max_grid_size_y, grid_size_y)

            # Interpolation from point cloud to uniform grid
            input_u_sd_grid = []
            point_cloud = input_xy   # (n_x, dim)
            point_cloud_val = np.transpose(input_u, (1, 2, 3, 0))  # (n_x, T, n_c, n_total)  
            interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
            interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)
            for i in range(n_subdomains):
                grid_size_x, grid_size_y = grid_sizes[i]
                bbox = bbox_sd[i]
                # Linear interpolation
                grid_x = np.linspace(bbox[0][0], bbox[0][1], num=grid_size_x)
                grid_y = np.linspace(bbox[1][0], bbox[1][1], num=grid_size_y)
                grid_x, grid_y = np.meshgrid(grid_x, grid_y)
                grid_val = interp_linear(grid_x, grid_y)  # (grid_x, grid_y, T, n_c, n_total)
                # Fill nan values
                nan_indices = np.isnan(grid_val)[..., 0, 0, 0]
                fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
                grid_val[nan_indices] = fill_vals
                # Resize to the same size via FFT-IFFT
                freq = np.fft.rfft2(grid_val, axes=(0, 1))
                s1_padded, s2_padded = max_grid_size_y, max_grid_size_x 
                if is_transposed[i]:
                    s1_padded, s2_padded = s2_padded, s1_padded
                square_freq = np.zeros((s1_padded, 
                    s2_padded // 2 + 1, T, n_c, n_total)) + 0j
                square_freq[:min(s1_padded//2, freq.shape[0]//2), 
                        :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                    freq[:min(s1_padded//2, freq.shape[0]//2), 
                        :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
                square_freq[-min(s1_padded//2, freq.shape[0]//2):, 
                        :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                    freq[-min(s1_padded//2, freq.shape[0]//2):,
                        :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
                grid_val = np.fft.irfft2(square_freq, 
                    s=(s1_padded, s2_padded), axes=(0, 1))
                if is_transposed[i]:
                    grid_val = np.transpose(grid_val, (1, 0, 2, 3, 4))   # (grid_x, grid_y, T, n_c, n_total)
                input_u_sd_grid.append(np.transpose(grid_val, (4, 0, 1, 2, 3))) # (n_subdomains, n_total, grid_x, grid_y, T, n_c,)
            input_u_sd_grid = np.transpose(np.array(input_u_sd_grid), (1, 2, 3, 4, 5, 0))  # (n_totol, grid_x,grid_y, T, n_c, n_subdomains )

            # Pad the point-cloud values of each subdomain to the same size
            # Mask is used to ignore padded zeros when calculating errors
            input_u_sd = np.zeros((n_total, 
                np.max([len(indices_sd[i]) for i in range(n_subdomains)]), T, n_c, n_subdomains))
            input_u_sd_mask = np.zeros((n_total, 
                np.max([len(indices_sd[i]) for i in range(n_subdomains)]), 1, 1, n_subdomains))  #(n_totol, maxlen_sd, 1,1, n_subdomains)
            for i in range(n_subdomains):
                input_u_sd[:, :len(indices_sd[i]), ..., i] = input_u[:, indices_sd[i], ...]
                input_u_sd_mask[:, :len(indices_sd[i]), ..., i] = 1.
            # print(input_u_sd.shape)
            # if SAVE_PREP:
            #     np.save(PATH_U_SD_G, input_u_sd_grid)
            #     np.save(PATH_U_SD, input_u_sd) 
            #     np.save(PATH_U_SD_M, input_u_sd_mask)
            # t2 = default_timer()
            # print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

            input_xy_sd = torch.from_numpy(input_xy_sd).float()
            input_xy_sd = input_xy_sd.unsqueeze(0)\
                .permute(0, 2, 1, 3)\
                .reshape(n_subdomains, -1, 1, 2)
                # shape: (n_subdomains, n_points_sd_padded, 1, 2)

            s1_padded, s2_padded = input_u_sd_grid.shape[1:3]
            input_u_sd_grid = input_u_sd_grid.reshape(s1_padded, s2_padded, T, n_c, n_subdomains).transpose((2, 0, 1, 4, 3)) #(T, s1_padded, s2_padded, n_subdomains, n_c)
            input_u_sd = input_u_sd.reshape(-1, T, n_c, n_subdomains).transpose((1, 0, 3, 2))  # (T, maxlen_sd, n_subdomains, n_c)
            input_u_sd_mask = torch.from_numpy(input_u_sd_mask.reshape(1,-1, 1, n_subdomains)).float()  #(maxlen_sd, 1, n_subdomains)
            input_u_sd_mask = input_u_sd_mask.repeat([T, 1, 1, 1]).permute((0, 1, 3, 2))  #（T, maxlen_sd, n_subdomains,1）
            
        return input_u_sd_grid, input_xy_sd, input_u_sd_mask, input_u_sd

# 单例模式， 只生成一次KD_tree
data_preprocessing = NUNO_Data_utils_2d()._data_preprocessing


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class NUFNO2d(nn.Module):
    def __init__(self, inputs_channel = 2, outputs_channel=None, modes1=12, modes2=12, width=20, pad = 2, initial_step=1, n_case_params = 5, n_subdomains = 6):
        super(NUFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_subdomains = n_subdomains
        self.padding = 9  # pad the domain if input is non-periodic
        in_channels = inputs_channel*n_subdomains*initial_step + 2 + n_case_params
        if outputs_channel is None:
            outputs_channel = inputs_channel

        self.out_channels = outputs_channel
        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, outputs_channel*n_subdomains)

    def forward(self, x, case_params, mask, grid, aux_data):
        """
        inputs:
            x:
            case_params:
            mask: (bs, maxlen_sd, n_subdomains, 1)
            grid: (bs, n_subdomains, maxlen_sd, dim )
            aux_data: () or (bs, size_x, size_y, n_subdomains, n_c)
        outputs:
            out: (bs, maxlen_sd, n_subdomains, n_c)
            x: (bs, size_x, size_y, n_subdomains, n_c)
        """
        if aux_data.numel() > 0:
            x = aux_data
        x = x.reshape(list(x.shape[:-2])+[-1])
        batch_size, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        grid1 = self.get_grid(x.shape, x.device)
        # print(x.shape, grid1.shape, case_params.shape, grid.shape)
        x = torch.cat((x, grid1, case_params), dim=-1)
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # print(grid.shape)
        input_xy_sd = grid.reshape(batch_size * self.n_subdomains, -1, 1, 2)  # (bs* n_subdomains, maxlen_sd, 1, dim), with (H, W)=(maxlen_sd, 1)
        x = x.reshape(batch_size, size_x, size_y, self.n_subdomains, self.out_channels)
        out = x.permute(0, 3, 1, 2, 4).reshape(-1, size_x, size_y, self.out_channels)  # (bs* n_subdomains, size_x, size_y, n_c ), with (H, W)=(size_x, size_y)

        u = F.grid_sample(input=out.permute(0, 3, 1, 2), grid=input_xy_sd, 
                    padding_mode='border', align_corners=False)  # (bs* n_subdomains, n_c, maxlen_sd, 1)

        out = u.squeeze(-1).permute(0, 2, 1)\
                    .reshape(batch_size, self.n_subdomains, -1,  self.out_channels)\
                    .permute(0, 2, 1, 3)  # (bs, maxlen_sd, n_subdomains, n_c)
        out = out*mask
        return out, x  # pred, aux_out

    def one_forward_step(self, x, case_params, mask,  grid, y, aux_data = torch.tensor(()), loss_fn=None, args= None):
        info = {}
        pred, x_next = self(x, case_params, mask, grid, aux_data)
        
        if loss_fn is not None:
            ## defined your specific loss calculations here
            loss = loss_fn(pred, y)
            return loss, pred, x_next, info
        else:
            #TODO: default loss_fn
            pass

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
