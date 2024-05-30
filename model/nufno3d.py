# from timeit import default_timer
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
# from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from util.utilities import *
from .tree import KDTree
from scipy.interpolate import LinearNDInterpolator, \
    NearestNDInterpolator, RegularGridInterpolator


def cal_grid_shape(tot_size, aspect_ratios: list[float]):
    '''
    Given the total size and (geometric) aspect ratio,
    output the grid shape.
    '''
    dim = len(aspect_ratios)
    shape = [None] * dim
    shape[0] = tot_size * np.prod([aspect_ratios[0] / \
        aspect_ratios[j] for j in range(1, dim)])
    shape[0] = shape[0] ** (1 / dim)
    for j in range(1, dim):
        shape[j] = aspect_ratios[j] / \
            aspect_ratios[0] * shape[0]
    shape = [max(int(np.round(l)), 2) for l in shape]
    return shape


################################################################
# 3d fourier layers
################################################################
class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv3d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class NUFNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, inputs_channel=3, outputs_channel=3, n_subdomains = 8, initial_step = 1, n_case_params = 5):
        super(NUFNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: (x_velocity, y_velocity, pressure) in [0, T)
        input shape: (batchsize, x=64, y=64, t=T, c=3)
        output: (x_velocity, y_velocity, pressure) in [T, 2T)
        output shape: (batchsize, x=64, y=64, t=T, c=3)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.n_subdomains = n_subdomains
        self.inputs_channel = inputs_channel
        in_channels = inputs_channel*initial_step*n_subdomains+3+n_case_params
        self.outputs_channel = outputs_channel
        self.p = nn.Linear(in_channels, self.width) # input channel is 6: (x_velocity, y_velocity, z_velocity) + 3 locations (u, v, w, x, y, z)
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.q = MLP(self.width, self.outputs_channel*n_subdomains, self.width * 4) # output channel is 3: (u, v, w)

    def forward(self, x, case_params, mask, grid):
        grid_samp = grid
        output = torch.zeros_like(x)
        batch_size = x.shape[0]
        x, case_params, bbox_sd, grid_shape, indices_sd, max_n_points_sd, order_sd, xyz, xyz_sd = self.data_interp(x, case_params, mask, grid)
        grid = self.get_grid(x.shape, x.device)
        # print(x.shape)
        x = torch.cat((x, grid, case_params), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic

        out = x.reshape(batch_size, 
                grid_shape[1], grid_shape[0], 
                grid_shape[2], self.outputs_channel, self.n_subdomains)
        # out = out.cpu().detach().numpy()

        # Interpolation (from grids to point cloud)
        out = out.permute(0, 5, 4, 1, 2, 3)\
            .reshape(-1, self.outputs_channel, 
                grid_shape[1], grid_shape[0], grid_shape[2])
            # Output shape: (batch * n_subdomains, output_dim
            #   s2, s1, s3)
        u = F.grid_sample(input=out, grid=xyz_sd, 
            padding_mode='border', align_corners=False)
            # Output shape: (batch * n_subdomains, output_dim, 
            #   n_points_sd_padded, 1, 1)
        out = u.squeeze(-1).squeeze(-1).permute(0, 2, 1)\
            .reshape(batch_size, self.n_subdomains, -1, self.outputs_channel)\
            .permute(0, 2, 3, 1)
            # Output shape: (batch_size, n_points_sd_padded, 
            #   output_dim, n_subdomains)

        for i in range(self.n_subdomains):
            for j in range(len(indices_sd[i])):
                output[:, indices_sd[i][j], :] = out[:, j, :, i]
        return output

    def data_interp(self, x, case_params, mask, grid):
        with torch.no_grad():
            n_subdomains = self.n_subdomains
            oversamp_ratio = 1.0
            xyz = grid[0].cpu().numpy()                 # shape: (19517, 3)
            input_point_cloud = x.detach().cpu().numpy()     # shape: (1000, 19517, 5)
            batch_size = x.shape[0]
            # T = batch_size

            # print("Start KD-Tree splitting...")
            # t1 = default_timer()
            point_cloud = xyz.tolist()
            # Use kd-tree to generate subdomain division
            tree= KDTree(
                point_cloud, dim=3, n_subdomains=n_subdomains, 
                n_blocks=8, return_indices=True
            )
            tree.solve()
            # Gather subdomain info
            bbox_sd = tree.get_subdomain_bounding_boxes()
            indices_sd = tree.get_subdomain_indices()
            # Pad the point cloud of each subdomain to the same size
            max_n_points_sd = np.max([len(indices_sd[i]) 
                for i in range(n_subdomains)])
            xyz_sd = np.zeros((1, max_n_points_sd, n_subdomains, 3))
            input_point_cloud_sd = np.zeros((batch_size, 
                max_n_points_sd, input_point_cloud.shape[-1], n_subdomains))
            # Mask is used to ignore padded zeros when calculating errors
            input_u_sd_mask = np.zeros((1, max_n_points_sd, 1, n_subdomains))
            # The maximum grid shape of subdomains
            grid_shape = [-1] * 3
                # (s1, s2, s3)
            # The new coordinate order of each subdomain
            # (after long side alignment)
            order_sd = []
            for i in range(n_subdomains):
                # Normalize to [-1, 1]
                _xyz = xyz[indices_sd[i], :]
                _min, _max = np.min(_xyz, axis=0, keepdims=True), \
                    np.max(_xyz, axis=0, keepdims=True)
                _xyz = (_xyz - _min) / (_max - _min) * 2 - 1
                # Long side alignment
                bbox = bbox_sd[i]
                scales = [bbox[j][1] - bbox[j][0] for j in range(3)]
                order = np.argsort(scales)
                _xyz = _xyz[:, order]
                order_sd.append(order.tolist())
                # Calculate the grid shape
                _grid_shape = cal_grid_shape(
                    oversamp_ratio * len(indices_sd[i]), scales)
                _grid_shape.sort()
                grid_shape = np.maximum(grid_shape, _grid_shape)
                # Applying
                xyz_sd[0, :len(indices_sd[i]), i, :] = _xyz
                input_point_cloud_sd[:, :len(indices_sd[i]), :, i] = \
                    input_point_cloud[:, indices_sd[i], :]
                input_u_sd_mask[0, :len(indices_sd[i]), 0, i] = 1.
            # print(grid_shape)
            grid_shape = np.array(grid_shape)
            # t2 = default_timer()
            # print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

            # Interpolation from point cloud to uniform grid
            # t1 = default_timer()
            print("Start interpolation...")
            input_sd_grid = []
            point_cloud = xyz
            point_cloud_val = np.transpose(input_point_cloud, (1, 2, 0)) 
            interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
            interp_nearest = NearestNDInterpolator(point_cloud, point_cloud_val)
            for i in range(n_subdomains):
                bbox = bbox_sd[i]
                _grid_shape = grid_shape[np.argsort(order_sd[i])]
                # print(_grid_shape)
                # Linear interpolation
                grid_x = np.linspace(bbox[0][0], bbox[0][1], 
                    num=_grid_shape[0])
                grid_y = np.linspace(bbox[1][0], bbox[1][1], 
                    num=_grid_shape[1])
                grid_z = np.linspace(bbox[2][0], bbox[2][1], 
                    num=_grid_shape[2])
                grid_x, grid_y, grid_z = np.meshgrid(
                    grid_x, grid_y, grid_z, indexing='ij')
                grid_val = interp_linear(grid_x, grid_y, grid_z)
                # Fill nan values
                nan_indices = np.isnan(grid_val)[..., 0, 0]
                fill_vals = interp_nearest(
                    np.stack((
                        grid_x[nan_indices], grid_y[nan_indices],
                        grid_z[nan_indices]), axis=1))
                grid_val[nan_indices] = fill_vals
                # Long size alignment
                grid_val = np.transpose(grid_val, 
                    order_sd[i] + [3, 4])
                input_sd_grid.append(np.transpose(grid_val, (4, 0, 1, 2, 3)))
            # Convert indexing to 'xy'
            input_sd_grid = np.transpose(
                np.array(input_sd_grid), (1, 3, 2, 4, 5, 0))
            # print(input_sd_grid.shape)

            # t2 = default_timer()
            # print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

        xyz_sd = torch.from_numpy(xyz_sd).cuda().float()
        xyz_sd = xyz_sd.repeat([batch_size, 1, 1, 1])\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size*n_subdomains, -1, 1, 1, 3)
                # shape: (batch * n_subdomains, n_points_sd_padded, 1, 1, 3)
        input_point_cloud_sd = torch.from_numpy(input_point_cloud_sd).float()
                # shape: (ntotal, n_points_sd_padded, output_dim, n_subdomains)
        input_u_sd_mask = torch.from_numpy(input_u_sd_mask).cuda().float()
                # shape: (1, n_points_sd_padded, 1, n_subdomains)
        input_sd_grid = torch.from_numpy(input_sd_grid).float()
                # shape: (n_total, s2, s1, s3, input_dim + output_dim, n_subdomains)

        train_a_sd_grid = input_sd_grid.\
                reshape(batch_size, grid_shape[1], 
                    grid_shape[0], grid_shape[2], -1).cuda()

        case_params = case_params[:, 0, :].reshape(batch_size, 1, 1, 1, -1).repeat([1, grid_shape[1], grid_shape[0], grid_shape[2], 1])
        return train_a_sd_grid, case_params, bbox_sd, grid_shape, indices_sd, max_n_points_sd, order_sd, xyz, xyz_sd
        

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
