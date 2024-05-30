# Codes for section: Results on Darcy Flow Equation

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
from .integral_operators import *
import operator
from functools import reduce
from functools import partial
from typing import Optional

from timeit import default_timer

class UNO2d(nn.Module):
    """
        The overall network. It contains 7 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 7 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: Velocity field of the current frame (may include pressure field), the case parameters(reshape as (B, x, y, p)), grid and mask
        input shape: (batchsize, x=S, y=S, c + p + 1 + 2) 
        output: the solution of the next timesteps
        output shape: (B, x, y, c)
        Here SxS is the spatial resolution
        in_channels = c + p + 1 + 2, p for case parameters, 1 for mask, 2 for grid
        with = uplifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator
        """
    
    def __init__(self,in_channels, out_channels, width,pad = 0, factor = 3/4, n_case_params = 5):
        super(UNO2d, self).__init__()

        # +1 for mask, +2 for grid
        self.in_channels = in_channels + 1 + 2 + n_case_params # input channel
        self.out_channels = out_channels
        self.width = width 
        self.factor = factor
        self.padding = pad  

        self.fc = nn.Linear(self.in_channels, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_2D(self.width, 2*factor*self.width,48, 48, 22, 22)

        self.L1 = OperatorBlock_2D(2*factor*self.width, 4*factor*self.width, 32, 32, 14,14)

        self.L2 = OperatorBlock_2D(4*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L3 = OperatorBlock_2D(8*factor*self.width, 8*factor*self.width, 16, 16,6,6)
        
        self.L4 = OperatorBlock_2D(8*factor*self.width, 4*factor*self.width, 32, 32,6,6)

        self.L5 = OperatorBlock_2D(8*factor*self.width, 2*factor*self.width, 48, 48,14,14)

        self.L6 = OperatorBlock_2D(4*factor*self.width, self.width, 64, 64,22,22) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, self.out_channels)

    def forward(self, x, case_params, mask, grid):
        x = torch.cat((x, mask, grid, case_params), dim=-1)
         
        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 3, 1, 2)
        
        
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding])
        
        D1,D2 = x_fc0.shape[-2],x_fc0.shape[-1]
        
        x_c0 = self.L0(x_fc0,int(D1*self.factor),int(D2*self.factor))
        x_c1 = self.L1(x_c0 ,D1//2,D2//2)

        x_c2 = self.L2(x_c1 ,D1//4,D2//4)        
        x_c3 = self.L3(x_c2,D1//4,D2//4)
        x_c4 = self.L4(x_c3,D1//2,D2//2)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)
        x_c5 = self.L5(x_c4,int(D1*self.factor),int(D2*self.factor))
        x_c5 = torch.cat([x_c5, x_c0], dim=1)
        x_c6 = self.L6(x_c5,D1,D2)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., :-self.padding, :-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 1)
        
        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_out = self.fc2(x_fc1)
        
        return x_out * mask
    
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
    
class UNO3d(nn.Module):
    """
        The overall network. It contains 7 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 7 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: Velocity field of the current frame (may include pressure field), the case parameters(reshape as (B, x, y, p)), grid and mask
        input shape: (batchsize, x=S, y=S, c + p + 1 + 2) 
        output: the solution of the next timesteps
        output shape: (B, x, y, c)
        Here SxS is the spatial resolution
        in_channels = c + p + 1 + 2, p for case parameters, 1 for mask, 2 for grid
        with = uplifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator
        """
    
    def __init__(self,in_channels, out_channels, width,pad = 0, factor = 3/4, n_case_params = 5):
        super(UNO3d, self).__init__()

        """
        The overall network. It contains 7 integral operator.
        1. Lift the input to the desire channel dimension by  self.fc, self.fc0 .
        2. 7 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the first 10 timesteps (u(1), ..., u(10)).
        input shape: (batchsize, x=S, y=S, t=10)
        output: the solution of the next timesteps
        output shape: (batchsize, x=S, y=S, t=1)
        Here SxS is the spatial resolution
        in_width = 12 (10 input time steps + (x,y) location)
        with = uplifting dimension
        pad = padding the domian for non-periodic input
        factor = factor for scaling up/down the co-domain dimension at each integral operator
        """
        # +1 for mask, +3 for grid
        self.in_channels = in_channels + 1 + 3 + n_case_params # input channel
        self.out_channels = out_channels
        self.width = width 
        self.factor = factor
        self.padding = pad  

        self.fc = nn.Linear(self.in_channels, self.width//2)

        self.fc0 = nn.Linear(self.width//2, self.width) # input channel is 3: (a(x, y), x, y)

        self.L0 = OperatorBlock_3D(self.width, 2*factor*self.width,32, 32, 32, 8, 8, 8)

        self.L1 = OperatorBlock_3D(2*factor*self.width, 4*factor*self.width, 16, 16, 16, 4,4,4)

        self.L2 = OperatorBlock_3D(4*factor*self.width, 8*factor*self.width, 8, 8, 8,2,2,2)
        
        self.L3 = OperatorBlock_3D(8*factor*self.width, 8*factor*self.width, 8, 8,8,2,2,2)
        
        self.L4 = OperatorBlock_3D(8*factor*self.width, 4*factor*self.width, 16, 16, 16,2,2,2)

        self.L5 = OperatorBlock_3D(8*factor*self.width, 2*factor*self.width, 32, 32,32,4,4,4)

        self.L6 = OperatorBlock_3D(4*factor*self.width, self.width, 64, 64,64,8,8,8) # will be reshaped


        self.fc1 = nn.Linear(2*self.width, 3*self.width)
        self.fc2 = nn.Linear(3*self.width + self.width//2, self.out_channels)

    def forward(self, x, case_params, mask, grid):
        x = torch.cat((x, mask, grid, case_params), dim=-1)

        x_fc = self.fc(x)
        x_fc = F.gelu(x_fc)

        x_fc0 = self.fc0(x_fc)
        x_fc0 = F.gelu(x_fc0)
        
        x_fc0 = x_fc0.permute(0, 4, 1, 2, 3)
        x_fc0 = F.pad(x_fc0, [self.padding,self.padding, self.padding,self.padding, self.padding,self.padding])
        
        D1,D2,D3 = x_fc0.shape[-3],x_fc0.shape[-2],x_fc0.shape[-1]

        x_c0 = self.L0(x_fc0,D1//2,D2//2,D3//2)

        x_c1 = self.L1(x_c0,D1//4,D2//4,D3//4)


        x_c2 = self.L2(x_c1,D1//8,D2//8,D3//8)

        
        x_c3 = self.L3(x_c2,D1//8,D2//8,D3//8)


        x_c4 = self.L4(x_c3 ,D1//4,D2//4,D3//4)
        x_c4 = torch.cat([x_c4, x_c1], dim=1)

        x_c5 = self.L5(x_c4 ,D1//2,D2//2,D3//2)
        x_c5 = torch.cat([x_c5, x_c0], dim=1)

        x_c6 = self.L6(x_c5,D1,D2,D3)
        x_c6 = torch.cat([x_c6, x_fc0], dim=1)

        if self.padding!=0:
            x_c6 = x_c6[..., self.padding:-self.padding, self.padding:-self.padding, self.padding:-self.padding]

        x_c6 = x_c6.permute(0, 2, 3, 4, 1)

        x_fc1 = self.fc1(x_c6)
        x_fc1 = F.gelu(x_fc1)
        
        x_fc1 = torch.cat([x_fc1, x_fc], dim=-1)
        x_out = self.fc2(x_fc1)
        
        return x_out
    
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
