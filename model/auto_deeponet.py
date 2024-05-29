"""
Contains the implementation of autoregressive DeepONet
"""

from itertools import product
from typing import List, Optional

import torch
from torch import nn, Tensor


class AutoDeepONet(nn.Module):
    """
    Auto-regressive DeepONet for CFD.

    Our task is different from the one that the original DeepONet. In the
    original DeepONet, the input function (input to the branch net)
    is the initial condition (IC), but here, we have a fixed (zero) IC.
    Instead, we have different boundary conditions (BCs), but we also
    want the model to predict the next time step given the current time step.
    Ideally, we should have two different branch nets, one accepting the
    BCs, one accepting the current time step.

    Here, we assume that the current time step includes the information about
    BCs (which are the values on the bounds), so we just feed
    the current time step to one branch net.
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        out_channel: int,
        # num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        act_name="relu",
        act_norm: bool = False,
        act_on_output: bool = False,
        irregular_geometry: bool = False,
        autoregressive_mode = True
    ):
        """
        Args:
        - branch_dim: int, the dimension of the branch net input.
        - trunk_dim: int, the dimension of the trunk net input.
        """
        super().__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.out_channel = out_channel
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output
        self.irregular_geometry = irregular_geometry
        self.autoregressive_mode = autoregressive_mode

        act_fn = get_act_fn(act_name, act_norm)
        self.branch_dims = [branch_dim] + [width] * (branch_depth-1)+[out_channel*width]
        self.trunk_dims = [trunk_dim] + [width] * (trunk_depth-1)+[out_channel*width]
        self.branch_net = Ffn(
            self.branch_dims,
            act_fn=act_fn,
            act_on_output=act_on_output,
        )
        self.trunk_net = Ffn(self.trunk_dims, act_fn=act_fn)
        self.bias = nn.Parameter(torch.zeros(1))  # type: ignore

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        query_point: Optional[Tensor] = None,
    ):
        """

        ### Args
        - inputs: (b, h, w, c) for grid or (b, nx, c) for points cloud 
        - case_params: (b, p)
        - query_point: (k, 2), k is the number of query points, each is
            an (x, y) coordinate.
        - mask: (b, h, w, 1) for grid or (b, nx, 1) for points cloud 

        ### Returns
            Output: Tensor, if query_points is not None, the shape is (b, k).
                Else, the shape is (b, h, w, c).

        Notations:
        - b: batch size
        - c: number of channels
        - h: height
        - w: width
        - p: number of case parameters
        - k: number of query points
        """
        if not self.irregular_geometry:
            batch_size, height, width, in_channel = inputs.shape
            # use full grid as query points
            query_idxs = torch.tensor(
                list(product(range(height), range(width))),
                dtype=torch.long,
                device=inputs.device,
            )  # (h * w, 2)
            query_point = (query_idxs.float() - 50) / 100  # (k, 2)
        else:
            batch_size, _, in_channel = inputs.shape
            query_point = query_point[0] # (b, k, 2) -> (k, 2)
        
        inputs = inputs * mask
        # Flatten
        flat_inputs = inputs.view(batch_size, -1)  # (b, h * w * c)

        # Simple prepend physical properties to the input field.
        flat_inputs = torch.cat([flat_inputs, case_params], dim=1)
        x_branch = self.branch_net(flat_inputs)

        n_query = query_point.shape[0]
        # Input to the trunk net
        x_trunk = self.trunk_net(query_point)  # (k, p)
        x_branch=x_branch.reshape([batch_size,1,self.out_channel,-1]) #(b, 1 ,c, p)
        x_trunk=x_trunk.reshape([1,n_query,self.out_channel,-1]) # (1, k, c, p)
        residuals = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (b, k, c)
        if self.autoregressive_mode:
            preds = inputs.view(batch_size, -1,self.out_channel) # (b, k, c)
            preds = preds + residuals
        else: # darcy 
            preds = residuals # (b, k, c)
        if not self.irregular_geometry: # recover to shape of grid
            preds = preds.view(-1, height, width, self.out_channel)  # (b, h, w, c)
        preds = preds* mask
        return preds
    
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

class AutoDeepONet_3d(nn.Module):
    """
    Auto-regressive DeepONet for CFD.

    Our task is different from the one that the original DeepONet. In the
    original DeepONet, the input function (input to the branch net)
    is the initial condition (IC), but here, we have a fixed (zero) IC.
    Instead, we have different boundary conditions (BCs), but we also
    want the model to predict the next time step given the current time step.
    Ideally, we should have two different branch nets, one accepting the
    BCs, one accepting the current time step.

    Here, we assume that the current time step includes the information about
    BCs (which are the values on the bounds), so we just feed
    the current time step to one branch net.
    """

    def __init__(
        self,
        branch_dim: int,
        trunk_dim: int,
        out_channel: int,
        # num_label_samples: int = 1000,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        width: int = 100,
        act_name="relu",
        act_norm: bool = False,
        act_on_output: bool = False,
        irregular_geometry: bool = False,
        autoregressive_mode = True
    ):
        """
        Args:
        - branch_dim: int, the dimension of the branch net input.
        - trunk_dim: int, the dimension of the trunk net input.
        """
        super().__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.out_channel = out_channel
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.act_name = act_name
        self.act_norm = act_norm
        self.act_on_output = act_on_output
        self.irregular_geometry = irregular_geometry
        self.autoregressive_mode = autoregressive_mode

        act_fn = get_act_fn(act_name, act_norm)
        self.branch_dims = [branch_dim] + [width] * (branch_depth-1)+[out_channel*width]
        self.trunk_dims = [trunk_dim] + [width] * (trunk_depth-1)+[out_channel*width]
        self.branch_net = Ffn(
            self.branch_dims,
            act_fn=act_fn,
            act_on_output=act_on_output,
        )
        self.trunk_net = Ffn(self.trunk_dims, act_fn=act_fn)
        self.bias = nn.Parameter(torch.zeros(1))  # type: ignore

    def forward(
        self,
        inputs: Tensor,
        case_params: Tensor,
        mask: Optional[Tensor] = None,
        query_point: Optional[Tensor] = None,
    ):
        """

        ### Args
        - inputs: (b, h, w, c) for grid or (b, nx, c) for points cloud 
        - case_params: (b, p)
        - query_point: (k, 2), k is the number of query points, each is
            an (x, y) coordinate.
        - mask: (b, h, w, 1) for grid or (b, nx, 1) for points cloud 

        ### Returns
            Output: Tensor, if query_points is not None, the shape is (b, k).
                Else, the shape is (b, h, w, c).

        Notations:
        - b: batch size
        - c: number of channels
        - h: height
        - w: width
        - p: number of case parameters
        - k: number of query points
        """
        if not self.irregular_geometry:
            batch_size, height, width, depth, in_channel = inputs.shape
            # use full grid as query points
            query_idxs = torch.tensor(
                list(product(range(height), range(width), range(depth))),
                dtype=torch.long,
                device=inputs.device,
            )  # (h * w * d, 3)
            query_point = (query_idxs.float() - 50) / 100  # (k, 3)
        else:
            batch_size, _, in_channel = inputs.shape
            query_point = query_point[0] # (b, k, 3) -> (k, 3)
        
        inputs = inputs * mask
        # Flatten
        flat_inputs = inputs.view(batch_size, -1)  # (b, h * w * c)

        # Simple prepend physical properties to the input field.
        flat_inputs = torch.cat([flat_inputs, case_params], dim=1)
        x_branch = self.branch_net(flat_inputs)

        n_query = query_point.shape[0]
        # Input to the trunk net
        x_trunk = self.trunk_net(query_point)  # (k, p)
        x_branch=x_branch.reshape([batch_size,1,self.out_channel,-1]) #(b, 1 ,c, p)
        x_trunk=x_trunk.reshape([1,n_query,self.out_channel,-1]) # (1, k, c, p)
        residuals = torch.sum(x_branch * x_trunk, dim=-1) + self.bias  # (b, k, c)
        if self.autoregressive_mode:
            preds = inputs.view(batch_size, -1,self.out_channel) # (b, k, c)
            preds = preds + residuals
        else: # darcy 
            preds = residuals # (b, k, c)
            
        if not self.irregular_geometry: # recover to shape of grid
            preds = preds.view(-1, height, width, depth, self.out_channel)  # (b, h, w, d, c)
        preds = preds* mask
        return preds
    
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



class Ffn(nn.Module):
    """
    A general fully connected multi-layer neural network.
    """

    def __init__(
        self, dims: list, act_fn: nn.Module, act_on_output: bool = False
    ):
        super().__init__()
        self.dims = dims

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn)
            # layers.append(NormAct(nn.Tanh()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if act_on_output:
            layers.append(act_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x

def get_act_fn(name: str, norm: bool = False) -> nn.Module:
    if name == "relu":
        fn = nn.ReLU()
    elif name == "tanh":
        fn = nn.Tanh()
    elif name == "gelu":
        fn = nn.GELU()
    elif name == "swish":
        fn = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    if norm:
        fn = NormAct(fn)
    return fn


class NormAct(nn.Module):
    """
    Normalized Activation Function.

    A wrapper around any activation function that normalizes the input
    before applying the activation function, and then transforms the
    output back to the original scale.
    """
    def __init__(self, act_fn: nn.Module):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (b, h, w)
        '''
        num_dims = len(x.shape)
        dims = tuple(range(1, num_dims))
        # find the mean and std of each example in the batch
        mean = x.mean(dim=dims, keepdim=True)
        std = x.std(dim=dims, keepdim=True)
        # normalize
        x = (x - mean) / std
        x = self.act_fn(x)
        # Transform back to the original scale
        x = x * std + mean
        return x