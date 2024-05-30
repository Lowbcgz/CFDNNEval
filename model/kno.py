import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# The structure of Auto-Encoder
class encoder_mlp(nn.Module):
    def __init__(self, t_len, width):
        super(encoder_mlp, self).__init__()
        self.layer = nn.Linear(t_len, width)
    def forward(self, x):
        x = self.layer(x)
        return x

class decoder_mlp(nn.Module):
    def __init__(self, t_len, width):
        super(decoder_mlp, self).__init__()
        self.layer = nn.Linear(width, t_len)
    def forward(self, x):
        x = self.layer(x)
        return x

class encoder_conv1d(nn.Module):
    def __init__(self, t_len, width):
        super(encoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(t_len, width,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class decoder_conv1d(nn.Module):
    def __init__(self, t_len, width):
        super(decoder_conv1d, self).__init__()
        self.layer = nn.Conv1d(width, t_len,1)
    def forward(self, x):
        x = x.permute([0,2,1])
        x = self.layer(x)
        x = x.permute([0,2,1])
        return x

class encoder_conv2d(nn.Module):
    def __init__(self, t_len, width):
        super(encoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(t_len, width,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x

class decoder_conv2d(nn.Module):
    def __init__(self, t_len, width):
        super(decoder_conv2d, self).__init__()
        self.layer = nn.Conv2d(width, t_len,1)
    def forward(self, x):
        x = x.permute([0,3,1,2])
        x = self.layer(x)
        x = x.permute([0,2,3,1])
        return x


class encoder_conv3d(nn.Module):
    def __init__(self, t_len, width):
        super(encoder_conv3d, self).__init__()
        self.layer = nn.Conv3d(t_len, width,1)
    def forward(self, x):
        x = x.permute([0,4,1,2,3])
        x = self.layer(x)
        x = x.permute([0,2,3,4,1])
        return x

class decoder_conv3d(nn.Module):
    def __init__(self, t_len, width):
        super(decoder_conv3d, self).__init__()
        self.layer = nn.Conv3d(width, t_len,1)
    def forward(self, x):
        x = x.permute([0,4,1,2,3])
        x = self.layer(x)
        x = x.permute([0,2,3,4,1])
        return x
    
# Koopman 1D structure
class Koopman_Operator1D(nn.Module):
    def __init__(self, width, modes_x = 16):
        super(Koopman_Operator1D, self).__init__()
        self.width = width
        self.scale = (1 / (width * width))
        self.modes_x = modes_x
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(width, width, self.modes_x, dtype=torch.cfloat))
    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x), (t, t+1, x) -> (batch, t+1, x)
        return torch.einsum("btx,tfx->bfx", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x] = self.time_marching(x_ft[:, :, :self.modes_x], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class KNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, autoencoder, width, n_case_params = 5, modes_x = 16, decompose = 4, linear_type = True, normalization = False):
        super(KNO1d, self).__init__()
        # Parameter
        self.width = width
        self.decompose = decompose
        # Layer Structure
        if autoencoder == 'MLP':
            self.enc = encoder_mlp(in_channels + n_case_params + 1, width) # +1 for mask.
            self.dec = decoder_mlp(out_channels, width)
        elif autoencoder == 'Conv':
            self.enc = encoder_conv1d(in_channels + n_case_params + 1, width)
            self.dec = decoder_conv1d(out_channels, width)
        else:
            raise("autoencoder type error! Please set it as 'MLP' or 'Conv'.")

        self.koopman_layer = Koopman_Operator1D(self.width, modes_x = modes_x)
        self.w0 = nn.Conv1d(width, width, 1)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(width)
    def forward(self, x, case_params, mask):
        x = torch.cat((x, mask, case_params), dim=-1)
        # Reconstruct
        x_reconstruct = self.enc(x)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        x_reconstruct = x_reconstruct * mask
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 2, 1)
        x_w = x
        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
        x = x.permute(0, 2, 1)
        x = self.dec(x) # Decoder
        x = x * mask
        return x, x_reconstruct

# Koopman 2D structure
class Koopman_Operator2D(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super(Koopman_Operator2D, self).__init__()
        self.width = width
        self.scale = (1 / (width * width))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(width, width, self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x,y ), (t, t+1, x,y) -> (batch, t+1, x,y)
        return torch.einsum("btxy,tfxy->bfxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfft2(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y] = self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class KNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, autoencoder, width, n_case_params = 5, modes_x = 10, modes_y = 10, decompose = 6, linear_type = True, normalization = False):
        super(KNO2d, self).__init__()
        # Parameter
        self.width = width
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        # Layer Structure
        if autoencoder == 'MLP':
            self.enc = encoder_mlp(in_channels + n_case_params + 1, width) # +1 for mask.
            self.dec = decoder_mlp(out_channels, width)
        elif autoencoder == 'Conv':
            self.enc = encoder_conv2d(in_channels + n_case_params + 1, width)
            self.dec = decoder_conv2d(out_channels, width)
        else:
            raise("autoencoder type error! Please set it as 'MLP' or 'Conv'.")
        
        self.koopman_layer = Koopman_Operator2D(self.width, self.modes_x, self.modes_y)
        self.w0 = nn.Conv2d(width, width, 1)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm2d(width)
    def forward(self, x, case_params, mask, grid):
        x = torch.cat((x, mask, case_params), dim=-1)
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 3, 1, 2)
        x_w = x
        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
        x = x.permute(0, 2, 3, 1)
        x = self.dec(x) # Decoder
        x = x * mask
        return x

    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        pred = self(x, case_params, mask, grid)

        x_reconstruct = torch.cat((x, mask, case_params), dim=-1)
        x_reconstruct = self.enc(x_reconstruct)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        x_reconstruct = x_reconstruct * mask
        
        if loss_fn is not None:
            ## defined your specific loss calculations here
            _batch = pred.size(0)
            loss_pred = loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
            loss_recons = loss_fn(x_reconstruct.reshape(_batch, -1), x.reshape(_batch, -1))
            loss = loss_pred * 5 + loss_recons * 0.5
            return loss, pred, info
        else:
            #TODO: default loss_fn
            pass


# Koopman 3D structure
class Koopman_Operator3D(nn.Module):
    def __init__(self, width, modes_x, modes_y, modes_z):
        super(Koopman_Operator3D, self).__init__()
        self.width = width
        self.scale = (1 / (width * width))
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.koopman_matrix = nn.Parameter(self.scale * torch.rand(width, width, self.modes_x, self.modes_y, self.modes_z, dtype=torch.cfloat))

    # Complex multiplication
    def time_marching(self, input, weights):
        # (batch, t, x,y ), (t, t+1, x,y) -> (batch, t+1, x,y)
        return torch.einsum("btxyz,tfxyz->bfxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Fourier Transform
        x_ft = torch.fft.rfftn(x)
        # Koopman Operator Time Marching
        out_ft = torch.zeros(x_ft.shape, dtype=torch.cfloat, device = x.device)
        out_ft[:, :, :self.modes_x, :self.modes_y, :self.modes_z] = \
            self.time_marching(x_ft[:, :, :self.modes_x, :self.modes_y, :self.modes_z], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, :self.modes_y, :self.modes_z] = \
            self.time_marching(x_ft[:, :, -self.modes_x:, :self.modes_y, :self.modes_z], self.koopman_matrix)
        out_ft[:, :, :self.modes_x, -self.modes_y:, :self.modes_z] = \
            self.time_marching(x_ft[:, :, :self.modes_x, -self.modes_y:, :self.modes_z], self.koopman_matrix)
        out_ft[:, :, -self.modes_x:, -self.modes_y:, :self.modes_z] = \
            self.time_marching(x_ft[:, :, -self.modes_x:, -self.modes_y:, :self.modes_z], self.koopman_matrix)
        #Inverse Fourier Transform
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
    
class KNO3d(nn.Module):
    def __init__(self, in_channels, out_channels, autoencoder, width, n_case_params = 5, modes_x = 10, modes_y = 10, modes_z = 10, decompose = 6, linear_type = True, normalization = False):
        super(KNO3d, self).__init__()
        # Parameter
        self.width = width
        self.decompose = decompose
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        # Layer Structure
        if autoencoder == 'MLP':
            self.enc = encoder_mlp(in_channels + n_case_params + 1, width) # +1 for mask.
            self.dec = decoder_mlp(out_channels, width)
        elif autoencoder == 'Conv':
            self.enc = encoder_conv3d(in_channels + n_case_params + 1, width)
            self.dec = decoder_conv3d(out_channels, width)
        else:
            raise("autoencoder type error! Please set it as 'MLP' or 'Conv'.")
        
        self.koopman_layer = Koopman_Operator3D(self.width, self.modes_x, self.modes_y, self.modes_z)
        self.w0 = nn.Conv3d(width, width, 1)
        self.linear_type = linear_type # If this variable is False, activate function will be worked after Koopman Matrix
        self.normalization = normalization
        if self.normalization:
            self.norm_layer = torch.nn.BatchNorm3d(width)
    def forward(self, x, case_params, mask, grid):
        x = torch.cat((x, mask, case_params), dim=-1)
        # Predict
        x = self.enc(x) # Encoder
        x = torch.tanh(x)
        x = x.permute(0, 4, 1, 2, 3)
        x_w = x
        for i in range(self.decompose):
            x1 = self.koopman_layer(x) # Koopman Operator
            if self.linear_type:
                x = x + x1
            else:
                x = torch.tanh(x + x1)
        if self.normalization:
            x = torch.tanh(self.norm_layer(self.w0(x_w)) + x)
        else:
            x = torch.tanh(self.w0(x_w) + x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.dec(x) # Decoder
        x = x * mask

        return x

    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        pred = self(x, case_params, mask, grid)

        x_reconstruct = torch.cat((x, mask, case_params), dim=-1)
        x_reconstruct = self.enc(x_reconstruct)
        x_reconstruct = torch.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)
        x_reconstruct = x_reconstruct * mask
        
        if loss_fn is not None:
            ## defined your specific loss calculations here
            _batch = pred.size(0)
            loss_pred = loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))
            loss_recons = loss_fn(x_reconstruct.reshape(_batch, -1), x.reshape(_batch, -1))
            loss = loss_pred * 5 + loss_recons * 0.5
            return loss, pred, info
        else:
            #TODO: default loss_fn
            pass