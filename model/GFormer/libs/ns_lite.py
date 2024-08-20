"""
(2+1)D Navier-Stokes equation + Galerkin Transformer
MIT license: Paper2394 authors, NeurIPS 2021 submission.
"""
import numpy as np
import torch
# from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import copy
import math
from collections import defaultdict

from .ft import WeightedL2Loss2d, WeightedL2Loss3d
from .layers import default, Identity
from .model import ADDITIONAL_ATTR, TransformerEncoderLayer, SimpleTransformerEncoderLayer, \
    PointwiseRegressor, SpectralRegressor, DownScaler,UpScaler, GCN, GAT
    
class FourierTransformer2DLite(nn.Module):
    def __init__(self, **kwargs):
        super(FourierTransformer2DLite, self).__init__()
        # defaultdict 对象，用于存储模型的配置参数
        # 接受 kwargs 参数，用于初始化配置参数
        
        self.config = defaultdict(lambda: None, **kwargs)
        self.out_dim = int(self.config['out_dim'])
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'

    def forward(self, x, case_params, mask, grid):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, n*n, node_feats)
        - pos: (batch_size, n*n, pos_dim)

        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        # process data
        node = torch.cat([x, case_params], dim=-1)
        edge = torch.tensor([1.0]) 
        # pos = torch.from_numpy(np.c_[grid[..., 0].ravel(), grid[..., 1].ravel()]) 
        grids_cpu = grid.cpu().numpy()
        pos_list = []
        for i in range(grids_cpu.shape[0]):
            grid_cpu = grids_cpu[i]
            pos = torch.from_numpy(np.c_[grid_cpu[..., 0].ravel(), grid_cpu[..., 1].ravel()]).to(node.device)
            pos_list.append(pos)
        pos = torch.stack(pos_list, dim=0)

        bsz = node.size(0)
        input_dim = node.size(-1)  # 2+p
        n_grid_x = grid.size(1)  
        n_grid_y = grid.size(2)

        node = torch.cat([node.view(bsz, -1, input_dim), pos],
                         dim=-1)  # (4, 64, 64, 2+p) --> (4, 64*64, 2+p+2)
        x = self.feat_extract(node, edge)  # --> (4, 64*64, 96/2)

        for encoder in self.encoder_layers:  # --> (4, 64*64, 96/2)
            x = encoder(x, pos)

        x = self.dpo(x)
        x = x.view(bsz, n_grid_x, n_grid_y, -1)  # --> (4, 64, 64, 96/2)

        x = self.regressor(x, grid=grid)  # --> (4, 64, 64, 1)

        return x

    def _initialize(self):
        self._get_feature()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])  # 模型就可以通过属性名来访问相应的配置参数

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)  # 若在配置参数中没有提供，则使用默认值来设置
        self.spacial_dim = default(self.spacial_dim, self.pos_dim)  # 空间维度的维度数
        self.spacial_fc = default(self.spacial_fc, False)  # 是否在空间维度上使用全连接层
        self.dropout = default(self.dropout, 0.05)
        
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        self.feat_extract = Identity(in_features=self.node_feats,
                                     out_features=self.n_hidden)  # Identity：恒等层

    def _get_encoder(self):
        # 用于初始化编码器
        encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                       n_head=self.n_head,
                                                       dim_feedforward=self.dim_feedforward,
                                                       layer_norm=self.layer_norm,
                                                       attention_type=self.attention_type,
                                                       attn_norm=self.attn_norm,
                                                       norm_type=self.norm_type,
                                                       xavier_init=self.xavier_init,
                                                       diagonal_weight=self.diagonal_weight,
                                                       dropout=self.encoder_dropout,
                                                       ffn_dropout=self.ffn_dropout,
                                                       pos_dim=self.pos_dim,
                                                       debug=self.debug)
        # 创建一个由多个编码器层组成的列表，并将该列表赋值给模型的 encoder_layers 属性
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                           n_hidden=self.n_hidden,
                                           freq_dim=self.freq_dim,
                                        #    out_dim=self.n_targets,
                                           out_dim=self.out_dim,
                                           num_spectral_layers=self.num_regressor_layers,
                                           modes=self.fourier_modes,
                                           spacial_dim=self.spacial_dim,
                                           spacial_fc=self.spacial_fc,
                                           dim_feedforward=self.freq_dim,
                                           activation=self.regressor_activation,
                                           dropout=self.decoder_dropout,
                                           )

    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        # 数据前处理
        # define diff
        y = y.cpu().numpy()
        targets_pad = np.pad(y, ((0, 0), (1, 1), (1, 1), (0, 0)),
                        'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        h = 1 / x.shape[1]  # step
        grad_x = (targets_pad[:, d:, s:-s] - targets_pad[:, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (targets_pad[:, s:-s, d:] - targets_pad[:, s:-s, :-d])/d  # (N, S_x, S_y, t)
        gradu = torch.from_numpy(np.stack([grad_x/h, grad_y/h], axis=-2)).to(x.device)
        y = torch.from_numpy(y).to(x.device)

        # one step forward
        pred = self(x, case_params, mask, grid)
 
        # 定义loss
        flow_name = None
        if case_params.shape[-1] == 0: #darcy
            flow_name = 'darcy'
        n_grid = x.shape[1]
        h = 1/n_grid  # 差分网格
        loss_fn = WeightedL2Loss2d(flow_name, regularizer=True, h=h, gamma=0.5)
        metric_func = WeightedL2Loss2d(flow_name, regularizer=False, h=h, )

        # 计算loss
        # 分通道计算loss并求和
        # pred, u: torch.Size([bs, x1, x2, 2 or 3]) 
        # targets_prime: torch.Size([bs, x1, x2, 2, 2 or 3])
        
        loss_u, reg_u, _, _ = loss_fn(pred, y, targets_prime=gradu)
        loss_u = loss_u + reg_u

        return loss_u, pred, info

class My_FourierTransformer2D(nn.Module):
    def __init__(self, **kwargs):
        super(My_FourierTransformer2D, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self.out_dim = int(self.config['out_dim'])
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'


    def forward(self, x, case_params, mask, grid):
        '''
        seq_len: n, number of grid points
        node_feats: number of features of the inputs
        pos_dim: dimension of the Euclidean space
        - node: (batch_size, n*n, node_feats)
        - pos: (batch_size, n*n, pos_dim)

        Remark:
        for classic Transformer: pos_dim = n_hidden = 512
        pos encodings is added to the latent representation
        '''
        # process data
        node = torch.cat([x, case_params], dim=-1)
        edge = torch.tensor([1.0]) 
        # pos = torch.from_numpy(np.c_[grid[..., 0].ravel(), grid[..., 1].ravel()]) 
        grids_cpu = grid.cpu().numpy()
        pos_list = []
        for i in range(grids_cpu.shape[0]):
            grid_cpu = grids_cpu[i]
            pos = torch.from_numpy(np.c_[grid_cpu[..., 0].ravel(), grid_cpu[..., 1].ravel()]).to(node.device)
            pos_list.append(pos)
        pos = torch.stack(pos_list, dim=0)
        weight = None
        boundary_value = None

        bsz = node.size(0)
        n_1 = int(node.size()[1])
        n_2 = int(node.size()[2])
        x_latent = []
        attn_weights = []
        # print('in model-------------------------------------------------------------')
        # print('x, pos:', node.shape, pos.shape)

        # if not self.downscaler_size:
        x = torch.cat(
                [node, pos.contiguous().view(bsz, n_1, n_2, -1)], dim=-1)
        
        # print('after downscale:', x.shape)
        x = x.view(bsz, -1, self.node_feats)
        # print('after view:', x.shape)
        x = self.feat_extract(x)  # 修改，不再传入edge
        # print('after extract:', x.shape, pos.shape)
        # x = self.dpo(x)

        for encoder in self.encoder_layers:
            if self.return_attn_weight and self.attention_type != 'official':
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            elif self.attention_type != 'official':
                x = encoder(x, pos, weight)
            else:
                out_dim = self.n_head*self.pos_dim + self.n_hidden
                x = x.view(bsz, -1, self.n_head, self.n_hidden//self.n_head).transpose(1, 2)
                x = torch.cat([pos.repeat([1, self.n_head, 1, 1]), x], dim=-1)
                x = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)
                x = encoder(x)
            if self.return_latent:
                x_latent.append(x.contiguous())
        # print('after encoder:', x.shape)

        x = x.view(bsz, n_1, n_2, self.n_hidden)
        # x = self.upscaler(x)
        # print('after upscale:', x.shape)

        if self.return_latent:
            x_latent.append(x.contiguous())

        x = self.dpo(x)
        # print('before regress:', x.shape, grid.shape)

        if self.return_latent:
            x, xr_latent = self.regressor(x, grid=grid)
            x_latent.append(xr_latent)
        else:
            x = self.regressor(x, grid=grid)
        # print('after regress:', x.shape)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.boundary_condition == 'dirichlet':
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)
            if boundary_value is not None:
                assert x.size() == boundary_value.size()
                x += boundary_value
        # print('out:', x.shape)

        return x

    def _initialize(self):
        self._get_feature()
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def cuda(self, device=None):
        self = super().cuda(device)
        if self.normalizer:
            self.normalizer = self.normalizer.cuda(device)
        return self

    def cpu(self):
        self = super().cpu()
        if self.normalizer:
            self.normalizer = self.normalizer.cpu()
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.normalizer:
            self.normalizer = self.normalizer.to(*args, **kwargs)
        return self

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    @staticmethod
    def _get_pos(pos, downsample):
        '''
        get the downscaled position in 2d
        '''
        bsz = pos.size(0)
        n_grid = pos.size(1)
        x, y = pos[..., 0], pos[..., 1]
        x = x.view(bsz, n_grid, n_grid)
        y = y.view(bsz, n_grid, n_grid)
        x = x[:, ::downsample, ::downsample].contiguous()
        y = y[:, ::downsample, ::downsample].contiguous()
        return torch.stack([x, y], dim=-1)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax']


    def _get_feature(self):
        self.feat_extract = Identity(in_features=self.node_feats,
                                     out_features=self.n_hidden)  # Identity：恒等层
        # if self.feat_extract_type == 'gcn' and self.num_feat_layers > 0:
            # self.feat_extract = GCN(node_feats=self.n_hidden,
        #                             edge_feats=self.edge_feats,
        #                             num_gcn_layers=self.num_feat_layers,
        #                             out_features=self.n_hidden,
        #                             activation=self.graph_activation,
        #                             raw_laplacian=self.raw_laplacian,
        #                             debug=self.debug,
        #                             )
        # elif self.feat_extract_type == 'gat' and self.num_feat_layers > 0:
        #     self.feat_extract = GAT(node_feats=self.n_hidden,
        #                             out_features=self.n_hidden,
        #                             num_gcn_layers=self.num_feat_layers,
        #                             activation=self.graph_activation,
        #                             debug=self.debug,
        #                             )
        # else:
        #     self.feat_extract = Identity()

    def _get_scaler(self):
        if self.downscaler_size:
            self.downscaler = DownScaler(in_dim=self.node_feats,
                                         out_dim=self.n_hidden,
                                         downsample_mode=self.downsample_mode,
                                         interp_size=self.downscaler_size,
                                         dropout=self.downscaler_dropout,
                                         activation_type=self.downscaler_activation)
        else:
            self.downscaler = Identity(in_features=self.node_feats+self.spacial_dim,
                                       out_features=self.n_hidden)
        if self.upscaler_size:
            self.upscaler = UpScaler(in_dim=self.n_hidden,
                                     out_dim=self.n_hidden,
                                     upsample_mode=self.upsample_mode,
                                     interp_size=self.upscaler_size,
                                     dropout=self.upscaler_dropout,
                                     activation_type=self.upscaler_activation)
        else:
            self.upscaler = Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           norm_eps=self.norm_eps,
                                                           debug=self.debug)
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden+self.pos_dim*self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.out_dim,  # 修改
                                                num_layers=self.num_regressor_layers,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                return_latent=self.return_latent,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft2':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.out_dim,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent,
                                               debug=self.debug
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")
    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        # diff
        y = y.cpu().numpy()
        targets_pad = np.pad(y, ((0, 0), (1, 1), (1, 1), (0, 0)),
                        'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        h = 1 / x.shape[1]  # step
        grad_x = (targets_pad[:, d:, s:-s] - targets_pad[:, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (targets_pad[:, s:-s, d:] - targets_pad[:, s:-s, :-d])/d  # (N, S_x, S_y, t)
        gradu = torch.from_numpy(np.stack([grad_x/h, grad_y/h], axis=-2)).to(x.device)
        y = torch.from_numpy(y).to(x.device)

        # one step forward
        pred = self(x, case_params, mask, grid)
 
        # 定义loss
        flow_name = None
        if case_params.shape[-1] == 0: #darcy
            flow_name = 'darcy'
        n_grid = x.shape[1]
        h = 1/n_grid  # 差分网格
        loss_fn = WeightedL2Loss2d(flow_name, regularizer=True, h=h, gamma=0.5)
        metric_func = WeightedL2Loss2d(flow_name, regularizer=False, h=h, )

        # 计算loss
        # 分通道计算loss并求和
        # pred, u: torch.Size([bs, x1, x2, 2 or 3]) 
        # targets_prime: torch.Size([bs, x1, x2, 2, 2 or 3])
        
        loss_u, reg_u, _, _ = loss_fn(pred, y, targets_prime=gradu)
        loss_u = loss_u + reg_u

        return loss_u, pred, info

class My_FourierTransformer3D(nn.Module):
    def __init__(self, **kwargs):
        super(My_FourierTransformer3D, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self.out_dim = int(self.config['out_dim'])
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'


    def forward(self, x, case_params, mask, grid):
        '''
        - node: (batch_size, n, n, node_feats)
        - pos: (batch_size, n_s*n_s, pos_dim)
        - edge: (batch_size, n_s*n_s, n_s*n_s, edge_feats)
        - weight: (batch_size, n_s*n_s, n_s*n_s): mass matrix prefered
            or (batch_size, n_s*n_s) when mass matrices are not provided (lumped mass)
        - grid: (batch_size, n-2, n-2, 2) excluding boundary
        '''
        # process data
        node = torch.cat([x, case_params], dim=-1)
        edge = torch.tensor([1.0]) 
        # pos = torch.from_numpy(np.c_[grid[..., 0].ravel(), grid[..., 1].ravel()]) 
        grids_cpu = grid.cpu().numpy()
        pos_list = []
        for i in range(grids_cpu.shape[0]):
            grid_cpu = grids_cpu[i]
            pos = torch.from_numpy(np.c_[grid_cpu[..., 0].ravel(), grid_cpu[..., 1].ravel()]).to(node.device)
            pos_list.append(pos)
        pos = torch.stack(pos_list, dim=0)
        weight = None
        boundary_value = None



        bsz = node.size(0)
        n_s = math.ceil(pos.size(1)**(1/3))
        x_latent = []
        attn_weights = []
        # print('in model-------------------------------------------------------------')
        # print('pos.size(1), n_s', pos.size(1), n_s)
        # print('x, pos:', node.shape, pos.shape)

        # 1、合并坐标点
        node = torch.cat(
                    [node, pos.contiguous().view(bsz, n_s, n_s, n_s, -1)], dim=-1)
        x = node

        # 2、网格拉成向量
        x = x.view(bsz, -1, self.node_feats)
    
        # 3、特征提取
        # print('after view:', x.shape)
        x = self.feat_extract(x)  # 修改，不再传入edge

        # print('after extract:', x.shape)
        x = self.dpo(x)
        
        # 4、encoder
        for encoder in self.encoder_layers:
            if self.return_attn_weight and self.attention_type != 'official':
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            elif self.attention_type != 'official':
                x = encoder(x, pos, weight)
            else:
                out_dim = self.n_head*self.pos_dim + self.n_hidden
                x = x.view(bsz, -1, self.n_head, self.n_hidden//self.n_head).transpose(1, 2)
                x = torch.cat([pos.repeat([1, self.n_head, 1, 1]), x], dim=-1)
                x = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)
                x = encoder(x)
            if self.return_latent:
                x_latent.append(x.contiguous())
        # print('after encoder:', x.shape)
        
        # 5、view
        x = x.view(bsz, n_s, n_s, n_s, self.n_hidden)
        # x = self.upscaler(x)
        # print('after upscale:', x.shape)

        if self.return_latent:
            x_latent.append(x.contiguous())

        x = self.dpo(x)
        # print('before regress:', x.shape)

        # 6、regress
        if self.return_latent:
            x, xr_latent = self.regressor(x, grid=grid)
            x_latent.append(xr_latent)
        else:
            x = self.regressor(x, grid=grid)
        # print('after regress:', x.shape)

        if self.normalizer:
            x = self.normalizer.inverse_transform(x)
        
        # 流体没有dirichlet边界，直接注释
        # if self.boundary_condition == 'dirichlet':
        #     x = x[:, 1:-1, 1:-1].contiguous()
        #     x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)
        #     if boundary_value is not None:
        #         assert x.size() == boundary_value.size()
        #         x += boundary_value
        # print('out:', x.shape)

        return x

    def _initialize(self):
        self._get_feature()
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def cuda(self, device=None):
        self = super().cuda(device)
        if self.normalizer:
            self.normalizer = self.normalizer.cuda(device)
        return self

    def cpu(self):
        self = super().cpu()
        if self.normalizer:
            self.normalizer = self.normalizer.cpu()
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.normalizer:
            self.normalizer = self.normalizer.to(*args, **kwargs)
        return self

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    @staticmethod
    def _get_pos(pos, downsample):
        '''
        get the downscaled position in 2d
        '''
        bsz = pos.size(0)
        n_grid = pos.size(1)
        x, y = pos[..., 0], pos[..., 1]
        x = x.view(bsz, n_grid, n_grid)
        y = y.view(bsz, n_grid, n_grid)
        x = x[:, ::downsample, ::downsample].contiguous()
        y = y[:, ::downsample, ::downsample].contiguous()
        return torch.stack([x, y], dim=-1)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        self.feat_extract = Identity(in_features=self.node_feats,
                                     out_features=self.n_hidden)  # Identity：恒等层
        # if self.feat_extract_type == 'gcn' and self.num_feat_layers > 0:
        #     self.feat_extract = GCN(node_feats=self.n_hidden,
        #                             edge_feats=self.edge_feats,
        #                             num_gcn_layers=self.num_feat_layers,
        #                             out_features=self.n_hidden,
        #                             activation=self.graph_activation,
        #                             raw_laplacian=self.raw_laplacian,
        #                             debug=self.debug,
        #                             )
        # elif self.feat_extract_type == 'gat' and self.num_feat_layers > 0:
        #     self.feat_extract = GAT(node_feats=self.n_hidden,
        #                             out_features=self.n_hidden,
        #                             num_gcn_layers=self.num_feat_layers,
        #                             activation=self.graph_activation,
        #                             debug=self.debug,
        #                             )
        # else:
        #     self.feat_extract = Identity()

    def _get_scaler(self):
        if self.downscaler_size:
            self.downscaler = DownScaler(in_dim=self.node_feats,
                                         out_dim=self.n_hidden,
                                         downsample_mode=self.downsample_mode,
                                         interp_size=self.downscaler_size,
                                         dropout=self.downscaler_dropout,
                                         activation_type=self.downscaler_activation)
        else:
            self.downscaler = Identity(in_features=self.node_feats+self.spacial_dim,
                                       out_features=self.n_hidden)
        if self.upscaler_size:
            self.upscaler = UpScaler(in_dim=self.n_hidden,
                                     out_dim=self.n_hidden,
                                     upsample_mode=self.upsample_mode,
                                     interp_size=self.upscaler_size,
                                     dropout=self.upscaler_dropout,
                                     activation_type=self.upscaler_activation)
        else:
            self.upscaler = Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           norm_eps=self.norm_eps,
                                                           debug=self.debug)
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden+self.pos_dim*self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.out_dim,  # 修改-最后一个张量输出维度
                                                num_layers=self.num_regressor_layers,
                                                spacial_fc=self.spacial_fc,  # true
                                                spacial_dim=self.spacial_dim,  # 网格维度
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                return_latent=self.return_latent,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft2':  # true
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.out_dim,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent,
                                               debug=self.debug
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")
    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        # diff
        y = y.cpu().numpy()
        targets_pad = np.pad(y, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)),
                    'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        h = 1 / x.shape[1]  # step
        grad_x = (targets_pad[:, d:, s:-s, s:-s] - targets_pad[:, :-d, s:-s, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (targets_pad[:, s:-s, d:, s:-s] - targets_pad[:, s:-s, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_z = (targets_pad[:, s:-s, s:-s, d:] - targets_pad[:, s:-s, s:-s, :-d])/d  # (N, S_x, S_y, t)
        gradu = torch.from_numpy(np.stack([grad_x/h, grad_y/h, grad_z/h], axis=-2)).to(x.device)
        y = torch.from_numpy(y).to(x.device)

        # one step forward
        pred = self(x, case_params, mask, grid)
 
        # 定义loss
        flow_name = None
        n_grid = x.shape[1]
        h = 1/n_grid  # 差分网格
        loss_fn = WeightedL2Loss3d(regularizer=True, h=h, gamma=0.5)
        metric_func = WeightedL2Loss3d(regularizer=True, h=h)

        # 计算loss
        # 分通道计算loss并求和
        # pred, u: torch.Size([bs, x1, x2, 2 or 3]) 
        # targets_prime: torch.Size([bs, x1, x2, 2, 2 or 3])
        
        loss_u, reg_u, _, _ = loss_fn(pred, y, targets_prime=gradu)
        loss_u = loss_u + reg_u

        return loss_u, pred, info

class Darcy_FourierTransformer2D(nn.Module):
    # 加入darcy中的上下采样
    def __init__(self, out_dim, **kwargs):
        super(Darcy_FourierTransformer2D, self).__init__()
        self.config = defaultdict(lambda: None, **kwargs)
        self.out_dim = out_dim
        self._get_setting()
        self._initialize()
        self.__name__ = self.attention_type.capitalize() + 'Transformer2D'

    def forward(self, x, case_params, mask, grid):
        '''
        - node: (batch_size, n, n, node_feats)
        - pos: (batch_size, n_s*n_s, pos_dim)
        - edge: (batch_size, n_s*n_s, n_s*n_s, edge_feats)
        - weight: (batch_size, n_s*n_s, n_s*n_s): mass matrix prefered
            or (batch_size, n_s*n_s) when mass matrices are not provided (lumped mass)
        - grid: (batch_size, n-2, n-2, 2) excluding boundary
        '''
        # process data
        node = torch.cat([x, case_params], dim=-1)
        edge = torch.tensor([1.0]) 
        subsample_attn = 6
        n_grid_fine = (x.shape[1]-1) * 3 + 1 # 逆过去求一下他逻辑下的原始网格大小  66->33
        n_grid = int(((n_grid_fine - 1)/subsample_attn))  # 采样后的网格大小
        grids_cpu = grid.cpu().numpy()
        pos_list = []
        for i in range(grids_cpu.shape[0]):
            grid_cpu = grids_cpu[i]
            indices_1 = np.linspace(0, grid_cpu.shape[0] - 1, n_grid).round().astype(int)
            indices_2 = np.linspace(0, grid_cpu.shape[1] - 1, n_grid).round().astype(int)
            # print('indices:', indices_1.shape, indices_2.shape)
            grid_downsampled = grid_cpu[indices_1, :, :][:, indices_2, :]  # [64, 64, 3] -> [31, 31, 3]
            # print('down:', grid_downsampled.shape)
            pos = np.c_[grid_downsampled[..., 0].ravel(), grid_downsampled[..., 1].ravel()] 
            pos = torch.from_numpy(pos)
            pos_list.append(pos)
        pos = torch.stack(pos_list, dim=0).to(node.device)
        weight = None
        boundary_value = None


        bsz = node.size(0)
        n_s = int(pos.size(1)**(0.5))
        x_latent = []
        attn_weights = []
        # print('in model-------------------------------------------------------------')
        # print('node, pos, n_s:', node.shape, pos.shape, n_s)
        if not self.downscaler_size:
            node = torch.cat(
                [node, pos.contiguous().view(bsz, n_s, n_s, -1)], dim=-1)
        x = self.downscaler(node)
        # print('after downscale:', x.shape, grid.shape)
        x = x.view(bsz, -1, self.n_hidden)
        # print('after view:', x.shape)

        x = self.feat_extract(x, edge)
        # print('after extract:', x.shape)
        x = self.dpo(x)

        for encoder in self.encoder_layers:
            if self.return_attn_weight and self.attention_type != 'official':
                x, attn_weight = encoder(x, pos, weight)
                attn_weights.append(attn_weight)
            elif self.attention_type != 'official':
                x = encoder(x, pos, weight)
            else:
                out_dim = self.n_head*self.pos_dim + self.n_hidden
                x = x.view(bsz, -1, self.n_head, self.n_hidden//self.n_head).transpose(1, 2)
                x = torch.cat([pos.repeat([1, self.n_head, 1, 1]), x], dim=-1)
                x = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)
                x = encoder(x)
            if self.return_latent:
                x_latent.append(x.contiguous())
        # print('after encoder:', x.shape)

        x = x.view(bsz, n_s, n_s, self.n_hidden)
        # print(self.upscaler_size)
        x = self.upscaler(x)
        # print('aftr upscale:', x.shape)

        if self.return_latent:
            x_latent.append(x.contiguous())

        x = self.dpo(x)
        # print('before regress:', x.shape, grid.shape)
        if self.return_latent:
            x, xr_latent = self.regressor(x, grid=grid)
            x_latent.append(xr_latent)
        else:
            x = self.regressor(x, grid=grid)
        # print('after regress:', x.shape)
        if self.normalizer:
            x = self.normalizer.inverse_transform(x)

        if self.boundary_condition == 'dirichlet':
            x = x[:, 1:-1, 1:-1].contiguous()
            x = F.pad(x, (0, 0, 1, 1, 1, 1), "constant", 0)
            if boundary_value is not None:
                assert x.size() == boundary_value.size()
                x += boundary_value
        # print('out:', x.shape)

        return x

    def _initialize(self):
        self._get_feature()
        self._get_scaler()
        self._get_encoder()
        self._get_regressor()
        self.config = dict(self.config)

    def cuda(self, device=None):
        self = super().cuda(device)
        if self.normalizer:
            self.normalizer = self.normalizer.cuda(device)
        return self

    def cpu(self):
        self = super().cpu()
        if self.normalizer:
            self.normalizer = self.normalizer.cpu()
        return self

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.normalizer:
            self.normalizer = self.normalizer.to(*args, **kwargs)
        return self

    def print_config(self):
        for a in self.config.keys():
            if not a.startswith('__'):
                print(f"{a}: \t", getattr(self, a))

    @staticmethod
    def _initialize_layer(layer, gain=1e-2):
        for param in layer.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=gain)
            else:
                constant_(param, 0)

    @staticmethod
    def _get_pos(pos, downsample):
        '''
        get the downscaled position in 2d
        '''
        bsz = pos.size(0)
        n_grid = pos.size(1)
        x, y = pos[..., 0], pos[..., 1]
        x = x.view(bsz, n_grid, n_grid)
        y = y.view(bsz, n_grid, n_grid)
        x = x[:, ::downsample, ::downsample].contiguous()
        y = y[:, ::downsample, ::downsample].contiguous()
        return torch.stack([x, y], dim=-1)

    def _get_setting(self):
        all_attr = list(self.config.keys()) + ADDITIONAL_ATTR
        for key in all_attr:
            setattr(self, key, self.config[key])

        self.dim_feedforward = default(self.dim_feedforward, 2*self.n_hidden)
        self.dropout = default(self.dropout, 0.05)
        self.dpo = nn.Dropout(self.dropout)
        if self.decoder_type == 'attention':
            self.num_encoder_layers += 1
        self.attention_types = ['fourier', 'integral', 'local', 'global',
                                'cosine', 'galerkin', 'linear', 'softmax']

    def _get_feature(self):
        # self.feat_extract = Identity(in_features=self.node_feats,
        #                              out_features=self.n_hidden)  # Identity：恒等层
        if self.feat_extract_type == 'gcn' and self.num_feat_layers > 0:
            self.feat_extract = GCN(node_feats=self.n_hidden,
                                    edge_feats=self.edge_feats,
                                    num_gcn_layers=self.num_feat_layers,
                                    out_features=self.n_hidden,
                                    activation=self.graph_activation,
                                    raw_laplacian=self.raw_laplacian,
                                    debug=self.debug,
                                    )
        elif self.feat_extract_type == 'gat' and self.num_feat_layers > 0:
            self.feat_extract = GAT(node_feats=self.n_hidden,
                                    out_features=self.n_hidden,
                                    num_gcn_layers=self.num_feat_layers,
                                    activation=self.graph_activation,
                                    debug=self.debug,
                                    )
        else:
            self.feat_extract = Identity()

    def _get_scaler(self):
        if self.downscaler_size:
            self.downscaler = DownScaler(in_dim=self.node_feats,
                                         out_dim=self.n_hidden,
                                         downsample_mode=self.downsample_mode,
                                         interp_size=self.downscaler_size,
                                         dropout=self.downscaler_dropout,
                                         activation_type=self.downscaler_activation)
        else:
            self.downscaler = Identity(in_features=self.node_feats+self.spacial_dim,
                                       out_features=self.n_hidden)
        if self.upscaler_size:
            self.upscaler = UpScaler(in_dim=self.n_hidden,
                                     out_dim=self.n_hidden,
                                     upsample_mode=self.upsample_mode,
                                     interp_size=self.upscaler_size,
                                     dropout=self.upscaler_dropout,
                                     activation_type=self.upscaler_activation)
        else:
            self.upscaler = Identity()

    def _get_encoder(self):
        if self.attention_type in self.attention_types:
            encoder_layer = SimpleTransformerEncoderLayer(d_model=self.n_hidden,
                                                           n_head=self.n_head,
                                                           attention_type=self.attention_type,
                                                           dim_feedforward=self.dim_feedforward,
                                                           layer_norm=self.layer_norm,
                                                           attn_norm=self.attn_norm,
                                                           batch_norm=self.batch_norm,
                                                           pos_dim=self.pos_dim,
                                                           xavier_init=self.xavier_init,
                                                           diagonal_weight=self.diagonal_weight,
                                                           symmetric_init=self.symmetric_init,
                                                           attn_weight=self.return_attn_weight,
                                                           dropout=self.encoder_dropout,
                                                           ffn_dropout=self.ffn_dropout,
                                                           norm_eps=self.norm_eps,
                                                           debug=self.debug)
        elif self.attention_type == 'official':
            encoder_layer = TransformerEncoderLayer(d_model=self.n_hidden+self.pos_dim*self.n_head,
                                                    nhead=self.n_head,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.encoder_dropout,
                                                    batch_first=True,
                                                    layer_norm_eps=self.norm_eps,
                                                    )
        else:
            raise NotImplementedError("encoder type not implemented.")
        self.encoder_layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])

    def _get_regressor(self):
        if self.decoder_type == 'pointwise':
            self.regressor = PointwiseRegressor(in_dim=self.n_hidden,
                                                n_hidden=self.n_hidden,
                                                out_dim=self.out_dim,  # 修改
                                                num_layers=self.num_regressor_layers,
                                                spacial_fc=self.spacial_fc,
                                                spacial_dim=self.spacial_dim,
                                                activation=self.regressor_activation,
                                                dropout=self.decoder_dropout,
                                                return_latent=self.return_latent,
                                                debug=self.debug)
        elif self.decoder_type == 'ifft2':
            self.regressor = SpectralRegressor(in_dim=self.n_hidden,
                                               n_hidden=self.freq_dim,
                                               freq_dim=self.freq_dim,
                                               out_dim=self.out_dim,
                                               num_spectral_layers=self.num_regressor_layers,
                                               modes=self.fourier_modes,
                                               spacial_dim=self.spacial_dim,
                                               spacial_fc=self.spacial_fc,
                                               activation=self.regressor_activation,
                                               last_activation=self.last_activation,
                                               dropout=self.decoder_dropout,
                                               return_latent=self.return_latent,
                                               debug=self.debug
                                               )
        else:
            raise NotImplementedError("Decoder type not implemented")
    def one_forward_step(self, x, case_params, mask,  grid, y, loss_fn=None, args= None):
        info = {}
        # diff
        y = y.cpu().numpy()
        targets_pad = np.pad(y, ((0, 0), (1, 1), (1, 1), (0, 0)),
                        'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        h = 1 / x.shape[1]  # step
        grad_x = (targets_pad[:, d:, s:-s] - targets_pad[:, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (targets_pad[:, s:-s, d:] - targets_pad[:, s:-s, :-d])/d  # (N, S_x, S_y, t)
        gradu = torch.from_numpy(np.stack([grad_x/h, grad_y/h], axis=-2)).to(x.device)
        y = torch.from_numpy(y).to(x.device)

        # one step forward
        pred = self(x, case_params, mask, grid)
 
        # 定义loss
        flow_name = None
        if case_params.shape[-1] == 0: #darcy
            flow_name = 'darcy'
        n_grid = x.shape[1]
        h = 1/n_grid  # 差分网格
        loss_fn = WeightedL2Loss2d(flow_name, regularizer=True, h=h, gamma=0.5)
        metric_func = WeightedL2Loss2d(flow_name, regularizer=False, h=h, )

        # 计算loss
        # 分通道计算loss并求和
        # pred, u: torch.Size([bs, x1, x2, 2 or 3]) 
        # targets_prime: torch.Size([bs, x1, x2, 2, 2 or 3])
        
        loss_u, reg_u, _, _ = loss_fn(pred, y, targets_prime=gradu)
        loss_u = loss_u + reg_u

        return loss_u, pred, info
