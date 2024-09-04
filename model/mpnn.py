# coding=utf-8
# This implementation is taken and modified from https://github.com/brandstetter-johannes/MP-Neural-PDE-Solvers/blob/master/experiments/models_gnn.py
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
from torch_geometric.nn import MessagePassing, InstanceNorm
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph


class Swish(nn.Module):
    """Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)
    

class GNN_Layer(MessagePassing):
    """Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 spatial_dim: int,
                 n_variables: int):
        """Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            spatial_dim (int): number of dimension of spatial domain  
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean') # node_dim: The axis along which to propagate. (default: -2)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        assert (spatial_dim == 1 or spatial_dim == 2 or spatial_dim == 3)

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + spatial_dim + n_variables, hidden_features), 
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features), 
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features), 
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features), 
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update

        

class MPNN2D(nn.Module):
    def __init__(self, 
                 neighbors: int = 1,
                 delta_t: float = 0.1,
                 hidden_features: int = 128,
                 hidden_layers: int = 6,
                 n_params: int = 5,
                 var_id: int = 0):
        """Modified Message Passing Neural PDE Solvers.
        Args:
            neighbors (int): The number of neighbors of each node in each direction on the graph. (default: 1)
            delta_t (float): Time step. (default: 0.1)
            hidden_features (int): The dimension of hidden feature. (default: 128)
            hidden_layers (int): The number of GNN layers. (default: 6)
            n_params (int): The number of case parameters. (default: 5)
            var_id (int): The variable subscript to be solved. (default: 0)
        """
        super().__init__()
        self.k = neighbors
        self.dt = delta_t
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.n_params = n_params
        self.var_id = var_id

        # encoder
        self.embedding_mlp = nn.Sequential(
            nn.Linear(1+2+n_params, self.hidden_features), # f([u, x, y, parmas])
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )

        # processor
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=1,
            spatial_dim=2,
            n_variables=n_params
            ) for _ in range(self.hidden_layers)))
        
        # decoder
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features // 2),
            Swish(),
            nn.Linear(self.hidden_features // 2, 1)
            )

    def forward(
        self,
        inputs,
        case_params,
        mask,
        grid
        ):
        """
        Args:
            inputs (Tensor): [bs, h, w, c]
            case_params (Tensor): [bs, h, w, num_case_params]
            mask (Tensor): [bs, h, w, 1]
            grid (Tensor): [bs, h, w, 2]

        Returns:
            out (Tensor): [bs, h, w, 1]
        """
        bs, h, w, c = inputs.shape
        if c > 1:
            inputs = inputs[..., self.var_id].unsqueeze(-1)
        
        # pre-process data
        graph = self.create_graph(inputs, case_params, grid)
        u = graph.x # [bs*num_points, 1]
        x_pos = graph.pos # [bs*num_points, 2]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*num_points]
        params = graph.params # [bs*num_points, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*num_points, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*num_points, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*num_points, 1]
        out = u + self.dt * diff # [bs*num_points, 1]

        return out.reshape([bs, h, w, -1])
    
    def create_graph(self, 
                     inputs, 
                     case_params,
                     grid):
        """
        Args:
            inputs (Tensor): [bs, h, w, 1]
            case_params (Tensor): [bs, h, w, num_case_params]
            grid (Tensor): [bs, h, w, 2]
        
        Returns:
            graph: graph data
        """
        device = inputs.device
        bs, h, w, c = inputs.shape
        x = torch.reshape(inputs, [-1, c])
        batch = torch.arange(bs).unsqueeze(-1).repeat(1, h*w).flatten().long().to(device) # [b*h*w]
        
        pos = torch.empty_like(grid)
        # normalize pos
        for i in range(grid.shape[-1]):
            max_value, _ = grid[..., i].reshape([bs, -1]).max(dim=-1)
            min_value, _ = grid[..., i].reshape([bs, -1]).min(dim=-1)
            max_value = max_value.unsqueeze(1).unsqueeze(2).repeat([1, h, w])
            min_value = min_value.unsqueeze(1).unsqueeze(2).repeat([1, h, w])
            pos[..., i] = (grid[..., i] - min_value) / (max_value - min_value)
        pos = torch.reshape(pos, [-1, 2])

        # compute edge index
        dx = 1 / min(h, w)
        X, Y = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="xy")
        logical_pos = torch.stack([X, Y], dim=-1).unsqueeze(0).repeat([bs, 1, 1, 1]) # [b, h, w, 2]
        edge_index = radius_graph(logical_pos.reshape([-1, 2]).to(device), r=np.sqrt(2)*self.k*dx, batch=batch, loop=False)

        graph = Data(x=x, edge_index=edge_index)
        graph.pos = pos
        graph.batch = batch
        graph.params = torch.reshape(case_params, [-1, self.n_params])
        graph.validate(raise_on_error=True)
    
        return graph
    
    def one_forward_step(self, x, case_params, mask, grid, y, loss_fn=None, args=None):
        """train step
        Args:
            x (Tensor): input with size [bs, h, w, c]
            case_params (Tensor): case parameters with size [bs, h, w, num_case_params]
            mask (Tensor): mask with size [bs, h, w, 1]
            grid (Tensor): grid with size [bs, h, w, 2]
            y (Tensor): label with size [bs, h, w, c]
            loss_fn (nn.Module): loss function
            args (dict): other arguments
        
        Returns:
            loss (Tensor): loss value
            out (Tensor): output with size [bs, h, w, 1]
            info (dict): other information
        """
        bs, h, w, c = x.shape
        if c > 1:
            x = x[..., self.var_id].unsqueeze(-1)
        y = y[..., self.var_id].unsqueeze(-1)
        
        # pre-process data
        graph = self.create_graph(x, case_params, grid)
        u = graph.x # [bs*num_points, 1]
        x_pos = graph.pos # [bs*num_points, 2]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*num_points]
        params = graph.params # [bs*num_points, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*num_points, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*num_points, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*num_points, 1]
        out = u + self.dt * diff # [bs*num_points, 1]

        # loss
        loss = loss_fn(out.reshape([bs, -1]), y.reshape([bs, -1]))

        return loss, out.reshape([bs, h, w, -1]), {}
    


class MPNN3D(nn.Module):
    def __init__(self, 
                 neighbors: int = 1, 
                 delta_t: float = 0.1, 
                 hidden_features: int = 128, 
                 hidden_layers: int = 6, 
                 n_params: int = 5, 
                 var_id: int = 0):
        """Modified Message Passing Neural PDE Solvers for 3D regular input
        Args:
            neighbors (int): The number of neighbors of each node in each direction on the graph. (default: 1)
            delta_t (float): Time step. (default: 0.1)
            hidden_features (int): The dimension of hidden feature. (default: 128)
            hidden_layers (int): The number of GNN layers. (default: 6)
            n_params (int): The number of case parameters. (default: 5)
            var_id (int): The variable subscript to be solved. (default: 0)
        """
        super().__init__()
        self.k = neighbors
        self.dt = delta_t
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.n_params = n_params
        self.var_id = var_id

        # encoder
        self.embedding_mlp = nn.Sequential(
            nn.Linear(1+3+n_params, self.hidden_features), # f([u, x, y, z, parmas])
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )

        # processor
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=1,
            spatial_dim=3,
            n_variables=n_params
            ) for _ in range(self.hidden_layers)))
        
        # decoder
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features // 2),
            Swish(),
            nn.Linear(self.hidden_features // 2, 1)
            )

    def forward(
        self,
        inputs,
        case_params,
        mask,
        grid
        ):
        """
        Args:
            inputs (Tensor): [bs, x, y, z, c]
            case_params (Tensor): [bs, x, y, z, num_case_params]
            mask (Tensor): [bs, x, y, z, 1]
            grid (Tensor): [bs, x, y, z, 3]

        Returns:
            out (Tensor): [bs, x, y, z, 1]
        """
        assert len(inputs.shape) == 5, "Expected input is tensor with shape like [bs, x, y, z, c]"
        bs, nx, ny, nz, c = inputs.shape
        if c > 1:
            inputs = inputs[..., self.var_id].unsqueeze(-1) # [bs, x, y, z, 1]
        
        # pre-process data
        graph = self.create_graph(inputs, case_params, grid)
        u = graph.x # [bs*num_points, 1]
        x_pos = graph.pos # [bs*num_points, 3]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*num_points]
        params = graph.params # [bs*num_points, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*num_points, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*num_points, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*num_points, 1]
        out = u + self.dt * diff # [bs*num_points, 1]

        return out.reshape([bs, nx, ny, nz, -1])
    
    def create_graph(self, 
                     inputs, 
                     case_params,
                     grid):
        """
        Args:
            inputs (Tensor): [bs, x, y, z, 1]
            case_params (Tensor): [bs, x, y, z, num_case_params]
            grid (Tensor): [bs, x, y, z, 3]
        
        Returns:
            graph: graph data
        """
        device = inputs.device
        bs, nx, ny, nz, c = inputs.shape
        x = torch.reshape(inputs, [-1, c])
        batch = torch.arange(bs).unsqueeze(-1).repeat(1, nx*ny*nz).flatten().long().to(device) # [bs*x*y*z]
        
        pos = torch.empty_like(grid)
        # normalize pos
        for i in range(grid.shape[-1]):
            max_value, _ = grid[..., i].reshape([bs, -1]).max(dim=-1) # [bs]
            min_value, _ = grid[..., i].reshape([bs, -1]).min(dim=-1)
            max_value = max_value.reshape([bs, 1, 1, 1]).repeat(1, nx, ny, nz)
            min_value = min_value.reshape([bs, 1, 1, 1]).repeat(1, nx, ny, nz)
            pos[..., i] = (grid[..., i] - min_value) / (max_value - min_value)
        pos = torch.reshape(pos, [-1, 3]) # [bs*x*y*z, 3]

        # compute edge index
        dx = 1 / min(nx, ny, nz)
        X, Y, Z = torch.meshgrid(torch.linspace(0, 1, nx), torch.linspace(0, 1, ny), torch.linspace(0, 1, nz),indexing="xy")
        logical_pos = torch.stack([X, Y, Z], dim=-1).unsqueeze(0).repeat([bs, 1, 1, 1, 1]) # [bs, x, y, z, 3]
        edge_index = radius_graph(logical_pos.reshape([-1, 3]).to(device), r=np.sqrt(2)*self.k*dx, batch=batch, loop=False)

        graph = Data(x=x, edge_index=edge_index)
        graph.pos = pos
        graph.batch = batch
        graph.params = torch.reshape(case_params, [-1, self.n_params])
        graph.validate(raise_on_error=True) 
    
        return graph
    
    def one_forward_step(self, x, case_params, mask, grid, y, loss_fn=None, args=None):
        """training step
        Args:
            x (Tensor): input with size [bs, x, y, z, c]
            case_params (Tensor): case parameters with size [bs, h, w, num_case_params]
            mask (Tensor): mask with size [bs, x, y, z, 1]
            grid (Tensor): grid with size [bs, x, y, z, 3]
            y (Tensor): label with size [bs, x, y, z, c]
            loss_fn (nn.Module): loss function
            args (dict): other arguments
        
        Returns:
            loss (Tensor): loss value
            out (Tensor): output with size [bs, x, y, z, 1]
            info (dict): other information
        """
        bs, nx, ny, nz, c = x.shape
        if c > 1:
            x = x[..., self.var_id].unsqueeze(-1)
        y = y[..., self.var_id].unsqueeze(-1)
        
        # pre-process data
        graph = self.create_graph(x, case_params, grid)
        u = graph.x # [bs*num_points, 1]
        x_pos = graph.pos # [bs*num_points, 3]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*num_points]
        params = graph.params # [bs*num_points, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*num_points, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*num_points, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*num_points, 1]
        out = u + self.dt * diff # [bs*num_points, 1]

        # loss
        loss = loss_fn(out.reshape([bs, -1]), y.reshape([bs, -1]))

        return loss, out.reshape([bs, nx, ny, nz, -1]), {}



class MPNNIrregular(nn.Module):
    def __init__(self, 
                 neighbors: int = 1,
                 delta_t: float = 0.1,
                 hidden_features: int = 128,
                 hidden_layers: int = 6,
                 n_params: int = 5,
                 var_id: int = 0,
                 spatial_dim: int = 2):
        super().__init__()
        self.k = neighbors
        self.dt = delta_t
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.n_params = n_params
        self.var_id = var_id
        self.spatial_dim = spatial_dim

        # encoder
        self.embedding_mlp = nn.Sequential(
            nn.Linear(1+spatial_dim+n_params, self.hidden_features), # f([u, x, y, parmas])
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
            )

        # processor
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=1,
            spatial_dim=spatial_dim,
            n_variables=n_params
            ) for _ in range(self.hidden_layers)))
        
        # decoder
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features // 2),
            Swish(),
            nn.Linear(self.hidden_features // 2, 1)
            )

    def forward(
        self,
        inputs,
        case_params,
        mask,
        grid
        ):
        """
        Args:
            inputs (Tensor): input with size [bs, nx, c]
            case_params (Tensor): case parameters with size [bs, nx,, num_case_params]
            mask (Tensor): mask with size [bs, nx, 1]
            grid (Tensor): grid with size [bs, nx, spatial_dim]

        Returns:
            out (Tensor): output with size [bs, nx, 1]
        """
        bs, nx, c = inputs.shape
        if c > 1:
            inputs = inputs[..., self.var_id].unsqueeze(-1)
        
        # pre-process data
        graph = self.create_graph(inputs, case_params, grid) # TODO
        u = graph.x # [bs*nx, 1]
        x_pos = graph.pos # [bs*nx, spatial_dim]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*nx]
        params = graph.params # [bs*nx, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*nx, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*nx, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*nx, 1]
        out = u + self.dt * diff # [bs*nx, 1]

        return out.reshape([bs, nx, -1])
    
    def create_graph(self, 
                     inputs, 
                     case_params,
                     grid):
        """
        Args:
            inputs (Tensor): [bs, nx, 1]
            case_params (Tensor): [bs, nx,, num_case_params]
            grid (Tensor): [bs, nx, spatial_dim]
        
        Returns:
            graph: graph data
        """
        device = inputs.device
        bs, nx, c = inputs.shape
        x = torch.reshape(inputs, [-1, c]) # [bs*nx, 1]
        batch = torch.arange(bs).unsqueeze(-1).repeat(1, nx).flatten().long().to(device) # [bs*nx]
        
        pos = torch.empty_like(grid) # [bs, nx, spatial_dim]
        # normalize pos
        for i in range(grid.shape[-1]):
            max_value, _ = grid[..., i].max(dim=-1) # [bs]
            min_value, _ = grid[..., i].min(dim=-1)
            max_value = max_value.unsqueeze(1).repeat([1, nx]) # [bs, nx]
            min_value = min_value.unsqueeze(1).repeat([1, nx])
            pos[..., i] = (grid[..., i] - min_value) / (max_value - min_value)
        pos = torch.reshape(pos, [-1, self.spatial_dim]) # [bs*nx, spatial_dim]

        edge_index = knn_graph(pos.to(device), k=self.k, batch=batch, loop=False)

        graph = Data(x=x, edge_index=edge_index)
        graph.pos = pos
        graph.batch = batch
        graph.params = torch.reshape(case_params, [-1, self.n_params])
        graph.validate(raise_on_error=True)
    
        return graph
    
    def one_forward_step(self, x, case_params, mask, grid, y, loss_fn=None, args=None):
        """train step
        Args:
            x (Tensor): input with size [bs, nx, c]
            case_params (Tensor): case parameters with size [bs, nx,, num_case_params]
            mask (Tensor): mask with size [bs, nx, 1]
            grid (Tensor): grid with size [bs, nx, spatial_dim]
            y (Tensor): label with size [bs, nx, c]
            loss_fn (nn.Module): loss function
            args (dict): other arguments

        Returns: 
            loss (Tensor): loss value
            out (Tensor): out with size [bs, nx, 1]
            info (dict): other information
        """
        bs, nx, c = x.shape
        if c > 1:
            x = x[..., self.var_id].unsqueeze(-1)
        y = y[..., self.var_id].unsqueeze(-1) # [bs, nx, 1]
        
        # pre-process data
        graph = self.create_graph(x, case_params, grid)
        u = graph.x # [bs*nx, 1]
        x_pos = graph.pos # [bs*nx, spatial_dim]
        edge_index = graph.edge_index # [2, num_edges]
        batch = graph.batch # [bs*nx]
        params = graph.params # [bs*nx, num_params]

        # encode
        node_input = torch.cat([u, x_pos, params], dim=-1)
        f = self.embedding_mlp(node_input) # [bs*nx, hidden_dim]
        # process
        for i in range(self.hidden_layers):
            f = self.gnn_layers[i](f, u, x_pos, params, edge_index, batch) # [bs*nx, hidden_dim]
        # decode
        diff = self.output_mlp(f) # [bs*nx, 1]
        out = u + self.dt * diff # [bs*nx, 1]

        # loss
        loss = loss_fn(out.reshape([bs, -1]), y.reshape([bs, -1]))

        return loss, out.reshape([bs, nx, -1]), {}