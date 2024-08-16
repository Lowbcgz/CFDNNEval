"""
(2+1)D Navier-Stokes equation + Galerkin Transformer
MIT license: Paper2394 authors, NeurIPS 2021 submission.
"""
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

from .utils import *
from .utils_ft import *
from .ft import *
from .layers import *
from .model import *
import h5py
import sklearn.model_selection as ms


class NavierStokesDatasetLite(Dataset):
    def __init__(self,
                 data_path=None,
                 train_data=True,
                 train_len=1024,
                 valid_len=200,
                 time_steps_input=10,
                 time_steps_output=10,
                 return_boundary=True,
                 random_state=1127802):
        '''
        PyTorch dataset overhauled for the Navier-Stokes turbulent
        regime data using the vorticity formulation from Li et al 2020
        https://github.com/zongyi-li/fourier_neural_operator

        Baseline:
        FNO2d+time marching network size: 926517
        Galerki transformer+2sc network size: 861617
        original grid size = 64*64, domain = (0,1)^2

        nodes: (N, n, n, T_0:T_1)
        pos: x, y coords flattened, (n*n, 2)
        grid: fine grid, x- and y- coords (n, n, 2)
        targets: solution u_h, (N, n, n, T_1:T_2)
        targets_grad: grad_h u_h, (N, n, n, 2, T_1:T_2)

        '''
        self.data_path = data_path
        self.n_grid = 64  # finest resolution along x-, y- dim
        self.h = 1/self.n_grid  # 网格步长
        self.train_data = train_data
        self.time_steps_input = time_steps_input
        self.time_steps_output = time_steps_output
        self.train_len = train_len
        self.valid_len = valid_len
        self.return_boundary = return_boundary
        self.random_state = random_state
        self.eps = 1e-8
        if self.data_path is not None:
            self._initialize()

    def __len__(self):
        return self.n_samples

    def _initialize(self):
        get_seed(self.random_state, printout=False)
        with timer(f"Loading {self.data_path.split('/')[-1]}"):
            data = h5py.File(self.data_path, mode='r')
            x = np.transpose(data['u'])  # x:(5000, 64, 64, 50)
            # (N, n, n, T_0:T_1)  前time_steps_input个时间步作为输入
            a = x[..., :self.time_steps_input]  
            # (N, n, n, T_1:T_2)  后面time_steps_output个时间步作为label
            u = x[...,
                  self.time_steps_input:self.time_steps_input+self.time_steps_output] 
            
            del data, x
            gc.collect()
        # 前train_len个作为训练集，后valid_len个作为验证集
        if self.train_data:
            a, u = a[:self.train_len], u[:self.train_len]
        else:
            a, u = a[-self.valid_len:], u[-self.valid_len:]
        self.n_samples = len(a)
        self.nodes, self.target, self.target_grad = self.get_data(a, u)

        x = np.linspace(0, 1, self.n_grid)
        y = np.linspace(0, 1, self.n_grid)
        x, y = np.meshgrid(x, y)
        # (n_grid, n_grid, 2)
        self.grid = np.stack([x, y], axis=-1)
        # (n_grid*n_grid, 2)
        self.pos = np.c_[x.ravel(), y.ravel()]

    def get_data(self, nodes, targets):
        targets_gradx, targets_grady = self.central_diff(targets, self.h)
        targets_grad = np.stack([targets_gradx, targets_grady], axis=-2)
        # targets = targets[..., None, :]  # (N, n, n, 1, T_1:T_2)
        # nodes = a[..., None, :]
        # 哪里来的None的维度？？？？？
        # targets_grad = [N, n-2, n-2, T_1:T_2]
        return nodes, targets, targets_grad

    @staticmethod
    def central_diff(x, h, padding=True):
        # 计算label的梯度
        # x: (batch, n, n, t)
        if padding:
            x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)),
                       'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s] - x[:, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (x[:, s:-s, d:] - x[:, s:-s, :-d])/d  # (N, S_x, S_y, t)

        return grad_x/h, grad_y/h

    def __getitem__(self, idx):
        # 输入数据与label，返回第idx条数据
        # nodes:(N, n, n, T_0:T_1)
        # target:(N, n, n, T_1:T_2)
        # target_grad:(N, n-2, n-2, T_1:T_2)
        # 网格数据，返回整个网格
        # grid:(n_grid, n_grid, 2)
        # pos:(n_grid*n_grid, 2)
        return dict(node=torch.from_numpy(self.nodes[idx]).float(),
                    pos=torch.from_numpy(self.pos).float(),
                    grid=torch.from_numpy(self.grid).float(),
                    target=torch.from_numpy(self.target[idx]).float(),
                    target_grad=torch.from_numpy(self.target_grad[idx]).float())  

    
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
        # data = process_data(x, case_params, mask,  grid, y)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # x, edge = data["node"].to(
        # device), data["edge"].to(device)
        # pos = data['pos_all'].to(device)  # 没有升降维，全网格
        # grid = data['grid'].to(device)
        # u, gradu = data["target"].to(device), data["target_grad"].to(device)
        # mask = data['mask'].to(device)

        # # print('in one step----------------------------------------------')
        # # print('x:', x.shape)
        # # print('y:', y.shape)

        # steps = u.size(-1)  # 3steps, every step with u,v,p  
        # steps_long = int(steps / 3)
        # if steps_long==0:
        #     steps_long = 1

        # 
        # out_ = self(x, edge, pos=pos, grid=grid)
        # pred = out_['preds']

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

def process_data(x, case_params, mask,  grid, y):
    
    inputs, labels, masks, case_params, grids = x.cpu(), y.cpu(), mask.cpu(), case_params.cpu(), grid.cpu()
    # subsample_attn = config['model']['subsample-attn']
    subsample_attn = 6
    # print('in process---------------------------------------------------------')
    # print('begin:', masks.shape)
    
    # 1. follow
    edge_features = torch.tensor([1.0])  # ？？？？
    mass_features = torch.tensor([1.0])
    x= inputs.shape[1]
    n_grid_fine = (x-1) * 3 + 1 # 逆过去求一下他逻辑下的原始网格大小  66->33
    n_grid = int(((n_grid_fine - 1)/subsample_attn))  # 采样后的网格大小
    h = 1/x

    # 2. 数据堆叠 --> nodes，targets
    nodes = torch.cat([inputs, case_params], dim=-1)
    # target:torch.Size([bs, 3, 64, 64, 64, 2]) --> [bs, 64, 64, 64, 6]
    # 将labels按照维度1堆叠在特征维度
    # print('label:', labels.shape)
    
    num_dim = inputs.shape  # [bs, x1, x2, 2] or [bs, x1, x2, x3, 4]
    # if config['model']['model_type'] == '3D':
    if num_dim == 5:
        if labels.ndim == 5:  # config['dataset']['multi_step_size']==1
            targets = labels
        else:
            targets = labels[:, 0, ...]
            for i in range(1, labels.shape[1]):
                targets = torch.cat([targets, labels[:, i, ...]], dim=-1)
                # print('target:', i, targets.shape)
        # 3. 计算梯度
        # padding
        # print('before padding------------------------')
        # print('target:', targets.shape)
        x = np.pad(targets, ((0, 0), (1, 1), (1, 1), (1, 1), (0, 0)),
                    'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s, s:-s] - x[:, :-d, s:-s, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (x[:, s:-s, d:, s:-s] - x[:, s:-s, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_z = (x[:, s:-s, s:-s, d:] - x[:, s:-s, s:-s, :-d])/d  # (N, S_x, S_y, t)
        targets_grad = np.stack([grad_x/h, grad_y/h, grad_z/h], axis=-2)

        # 4. 构建pos及pos_all
        # grids:[bs, n, n, n, 3]
        pos_list = []
        pos_all_list = []
        mask_list = []
        for i in range(grids.shape[0]):
            grid = grids[i]
            # 网格降维
            indices = np.linspace(0, grid.shape[0] - 1, n_grid).round().astype(int)
            grid_downsampled = grid[indices, :, :, :][:, indices, :, :][:, :, indices, :]  # [64, 64, 64, 3] -> [31, 31, 31, 3]
            pos = np.c_[grid_downsampled[..., 0].ravel(), grid_downsampled[..., 1].ravel(), grid_downsampled[..., 2].ravel()] 
            pos = torch.from_numpy(pos)
            pos_list.append(pos)
            # 原网格
            pos_all = torch.from_numpy(np.c_[grid[..., 0].ravel(), grid[..., 1].ravel()])  # 未降维的pos
            pos_all_list.append(pos_all)
            # mask
            mask = masks[i]  # (x, y, 1)
            # print('in process mask:', mask.shape)
            # print(mask.shape)
            if mask.ndim == 5:
                mask = mask.squeeze(-1).permute([1, 2, 3, 0]).repeat(1, 1, 1, labels.shape[-1])
            else:
                mask = mask.repeat(1, 1, 1, labels.shape[-1])
            mask_list.append(mask)
            
        pos = torch.stack(pos_list, dim=0)
        pos_all = torch.stack(pos_all_list, dim=0)
        mask = torch.stack(mask_list, dim=0)

    else:
        if labels.ndim == 4:  # config['dataset']['multi_step_size']==1
            targets = labels
        else:
            targets = labels[:, 0, ...]
            for i in range(1, labels.shape[1]):
                targets = torch.cat([targets, labels[:, i, ...]], dim=-1)
            # print('target:', i, targets.shape)
        # print('target:', targets.shape)
        # 3. 计算梯度  2d
        # padding  
        # print('before padding------------------------')
        # print('target:', targets.shape)
        x = np.pad(targets, ((0, 0), (1, 1), (1, 1), (0, 0)),
                        'constant', constant_values=0)
        d, s = 2, 1  # dilation and stride
        grad_x = (x[:, d:, s:-s] - x[:, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (x[:, s:-s, d:] - x[:, s:-s, :-d])/d  # (N, S_x, S_y, t)
        targets_grad = np.stack([grad_x/h, grad_y/h], axis=-2)

        # 4. 构建pos及pos_all 2d
        # grids:[bs, n, n, 3]
        pos_list = []
        pos_all_list = []
        mask_list = []
        for i in range(grids.shape[0]):
            grid = grids[i]
            # 网格降维
            # print('in process sata-------------------------------------------------------------')
            # print('grid:', grid.shape)
            indices_1 = np.linspace(0, grid.shape[0] - 1, n_grid).round().astype(int)
            indices_2 = np.linspace(0, grid.shape[1] - 1, n_grid).round().astype(int)
            # print('indices:', indices_1.shape, indices_2.shape)
            grid_downsampled = grid[indices_1, :, :][:, indices_2, :]  # [64, 64, 3] -> [31, 31, 3]
            # print('down:', grid_downsampled.shape)
            pos = np.c_[grid_downsampled[..., 0].ravel(), grid_downsampled[..., 1].ravel()] 
            pos = torch.from_numpy(pos)
            pos_list.append(pos)
            # 原网格
            pos_all = torch.from_numpy(np.c_[grid[..., 0].ravel(), grid[..., 1].ravel()])  # 未降维的pos
            pos_all_list.append(pos_all)
            # mask
            mask = masks[i]  # ([3, 66, 66, 1])
            if mask.ndim == 4:
                mask = mask.squeeze(-1).permute([1, 2, 0]).repeat(1, 1, labels.shape[-1])
            else:
                mask = mask.repeat(1, 1, labels.shape[-1])
            mask_list.append(mask)
            
        pos = torch.stack(pos_list, dim=0)
        pos_all = torch.stack(pos_all_list, dim=0)
        mask = torch.stack(mask_list, dim=0)

    return dict(node=nodes.float(),
                pos=pos.float(),
                pos_all=pos_all,
                grid=grids,
                edge=edge_features.float(),
                mass=mass_features.float(),
                target=targets,
                target_grad=torch.from_numpy(targets_grad),
                # case_id=case_ids,
                mask=mask)




def train_batch_ns(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    # 每走三步训练一次

    # 输入每一个batch的数据
    optimizer.zero_grad()
    x = data["node"].to(device)  # ([4, 64, 64, 7])  2+p
    pos, grid = data['pos'].to(device), data['grid'].to(device)  # [4, 4096, 2] [4, 64, 64, 2]
    u, gradu = data["target"].to(device), data["target_grad"].to(device)  # [4, 64, 64, 2*3] [4, 64, 64, 2, 2*3]
    # print('training_data-----------------------------------------------------------------------------------------------')
    # print('x, u:', x.shape, u.shape)
    pos_all = data['pos_all'].to(device)
    if config['model']['model_type'] in ['lite', 'ns']:
        pos = pos_all
    
    steps = u.size(-1)  # 6
    if steps==1:
        steps_long = 1
    else:
        steps_long = int(steps / 3)
    # print('in training--------------------------------------------------------------------------------------------------')
    # print('x, u:', x.shape, u.shape, pos.shape, grid.shape)
    # print('steps_long:', steps_long)
    loss_total = 0
    reg_total = 0

    u_preds = []

    for t in range(0, steps, steps_long):

        out_ = model(x, None, pos=pos, grid=grid)  # in:[4, 64, 64, 2+p] --> out:[4, 64, 64, 2]
        u_pred = out_['preds']  # torch.Size([4, 64, 64, 2])
        u_step = u[..., t:t+steps_long]
        gradu_step = gradu[..., t:t+steps_long]
        # print('train------------------------------------------------------------------------------------------------')
        # print('steps, steps_long, pred, u:', t, steps_long, u_pred.shape, u_step.shape)
        
        for i in range(steps_long):
            # steps_long=2:u, v              steps_long=3:u, v, p
            loss_u, reg_u, _, _ = loss_func(u_pred[..., i], u_step[..., i],
                                    targets_prime=gradu_step[..., i])
            loss_u = loss_u + reg_u
            loss_total += loss_u
            reg_total += reg_u.item()

        #     loss_total += loss_u
        #     reg_total += reg_u



        # # loss of u
        # loss_u, reg_u, _, _ = loss_func(u_pred[..., 0], u_step[..., 0],
        #                             targets_prime=gradu_step[..., 0])
        # loss_u = loss_u + reg_u
        # loss_total_u += loss_u
        # reg_total_u += reg_u.item()
        # # loss of v
        # loss_v, reg_v, _, _ = loss_func(u_pred[..., 1], u_step[..., 1],
        #                             targets_prime=gradu_step[..., 1])
        # loss_v = loss_v + reg_v
        # loss_total_v += loss_v
        # reg_total_v += reg_v.item()
        # # loss
        # loss_total = loss_total_u + loss_total_v
        # reg_total = reg_total_u + reg_total_v

        
        # x = torch.cat((x[..., 1:], u_step), dim=-1)  # 原变换
        x = torch.cat((u_step, x[..., steps_long:]), dim=-1)

        u_preds.append(u_pred)

    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    u_preds = torch.cat(u_preds, dim=-1).detach()



    return (loss_total.item()/steps, reg_total/steps), u_preds, None

def validate_epoch_ns(model, metric_func, valid_loader, device):
    # print('bug in valid')
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x = data["node"].to(device)
            u = data["target"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)
            # print('valid_data-----------------------------------------------------------------------------------------------')
            # print('x, u:', x.shape, u.shape)
            # print('pos, grid:', pos.shape, grid.shape)
            steps = u.size(-1)
            step_long = int(steps / 3)
            metric_val_step = 0

            for t in range(0, steps, step_long):
            
                out_ = model(x, None, pos=pos, grid=grid)
                u_pred = out_['preds']
                u_step = u[..., t:t+step_long]
                # print('valid------------------------------------------------------------------------------------------------')
                # print('steps, steps_long, pred, u:', t, step_long, u_pred.shape, u_step.shape)
                
                
                _, _, metric, _ = metric_func(u_pred, u_step)
                # x = torch.cat((x[..., 1:], u_pred), dim=-1)  # 原变换
                x = torch.cat((u_pred, x[..., step_long:]), dim=-1)
                # print('new x:', x.shape)
                # x:u+v+params
                # if t%2==0:
                #     x = torch.cat((u_pred, x[..., 1:]), dim=-1)  # 奇数步：新的u替换旧的u
                # else:
                #     x = torch.cat((x[..., 0].unsqueeze(-1), u_pred, x[..., 2:]), dim=-1)  # 偶数步：新的v替换旧的v
                metric_val_step += metric

        metric_val.append(metric_val_step/steps)
        # print('bug out of valid', metric_val)

    return dict(metric=np.mean(metric_val, axis=0))


def new_train_batch_ns(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99, ):
    # 每走三步训练一次

    optimizer.zero_grad()

    a, x, edge = data["coeff"].to(device), data["node"].to(
        device), data["edge"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    u, gradu = data["target"].to(device), data["target_grad"].to(device)
    mask = data['mask'].to(device)
    u = u * mask
    
    steps = u.size(-1)  # 3steps, every step with u,v,p
    if steps==1:
        steps_long = 1
    else:
        steps_long = int(steps / 3)
    # print('in train batch--------------------------------------------------------------------------------------------------')
    # print('x, u:', x.shape, u.shape, pos.shape, grid.shape)
    # print('steps_long:', steps_long)
    loss_total = 0
    reg_total = 0

    u_preds = []

    for t in range(0, steps, steps_long):
        out_ = model(x, edge, pos=pos, grid=grid)  # in:[4, 64, 64, 2+p] --> out:[4, 64, 64, 2]
        if isinstance(out_, dict):
            out = out_['preds']
        elif isinstance(out_, tuple):
            out = out_[0]
        # print('out:', out.shape)
        u_step = u[..., t:t+steps_long]
        gradu_step = gradu[..., t:t+steps_long]

        # u_pred:[4, 64, 64, 2], pred_grad:[], target:[4, 64, 64, 2]
        u_pred, pred_grad = out, out[..., steps_long:]
        # print('u_pred, target, pred_grad, gradu,a:',u_pred.shape, u_step.shape, pred_grad.shape, gradu.shape,a.shape )

        for i in range(steps_long):
            # steps_long=2:u, v              steps_long=3:u, v, p
            loss_u, reg_u, _, _ = loss_func(u_pred[..., i], u_step[..., i],pred_grad,
                                    targets_prime=gradu_step[..., i])
            loss_u = loss_u + reg_u
            loss_total += loss_u
            reg_total += reg_u.item()
      
        # x = torch.cat((x[..., 1:], u_step), dim=-1)  # 原变换
        x = torch.cat((u_step, x[..., steps_long:]), dim=-1)
        u_preds.append(u_pred)
    loss_total.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    u_preds = torch.cat(u_preds, dim=-1).detach()
    # print('loss.item():', loss_total.item())
    # print('loss:', loss_total)
    # print('loss_total.item()/steps:', loss_total.item()/steps)
    # print(reg_total/steps)

    return (loss_total.item()/steps, reg_total/steps), u_preds, None

def new_validate_epoch_ns(model, metric_func, valid_loader, device, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], plot_interval = 1):
    model.eval()
    metric_val = []

    # new
    val_l2 = 0
    val_l_inf = 0
    step = 0
    res_dict = {"cw_res":{},  # channel-wise
                "sw_res":{},}  # sample-wise
    # input_loss_dict={}
    # pred_loss_dict={}
    for name in metric_names:
        res_dict["cw_res"][name] = []
        res_dict["sw_res"][name] = []
        # input_loss_dict[name] = []
        # pred_loss_dict[name]=[]

    # ckpt_dir = output_dir + f"/ckpt-{epoch}"
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)

    for _, data in enumerate(valid_loader):
        step += 1  # new
        with torch.no_grad():
            x, edge = data["node"].to(device), data["edge"].to(device)
            u = data["target"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)
            mask = data['mask'].to(device)
            u = u * mask
            # print('valid_data-----------------------------------------------------------------------------------------------')
            # print('x, u:', x.shape, u.shape)
            # print('pos, grid:', pos.shape, grid.shape)
            steps = u.size(-1)
            if steps==1:
                step_long = 1
            else:
                step_long = int(steps / 3)
            metric_val_step = 0

            if getattr(valid_loader.dataset,"multi_step_size", 1)==1:
                # Model run
                # if case_params.shape[-1] == 0: #darcy
                #     case_params = case_params.reshape(0)
                out_ = model(x, edge, pos=pos, grid=grid)
                u_pred = out_['preds']
                u_step = u[..., t:t+step_long]
                _, _, metric, _ = metric_func(u_pred, u_step)
                # Loss calculation
                _batch = u_pred.size(0)

                # pred_loss_dict["MSE"].append(loss_fn(pred, y).cpu().detach())
                # pred_loss_dict["NMSE"].append((loss_fn(pred, y)/y.square().mean()).cpu().detach())
                # input_loss_dict["MSE"].append(loss_fn(x[...,:1], y[..., :1]).cpu().detach())
                # input_loss_dict["NMSE"].append((loss_fn(x[..., :1], y[..., :1])/y[..., :1].square().mean()).cpu().detach())

                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    cw, sw=metric_fn(pred, y)
                    res_dict["cw_res"][name].append(cw)
                    res_dict["sw_res"][name].append(sw)
                
                # if step % plot_interval == 0:
                #     image_dir = Path(ckpt_dir + "/images")
                #     if not os.path.exists(image_dir):
                #         os.makedirs(image_dir)
                #     plot_predictions(inp = x, label = y, pred = pred, out_dir=image_dir, step=step)

            

            for t in range(0, steps, step_long):
                out_ = model(x, edge, pos=pos, grid=grid)
                u_pred = out_['preds']
                u_step = u[..., t:t+step_long]
                # print('steps, steps_long, pred, u:', t, step_long, u_pred.shape, u_step.shape)
                
                _, _, metric, _ = metric_func(u_pred, u_step)
                x = torch.cat((u_pred, x[..., step_long:]), dim=-1)
                metric_val_step += metric
            try:
                metric_val.append(metric_val_step.item()/3)
            except:
                metric_val.append(metric_val_step/3)


        # print('bug out of valid', metric_val)

    return dict(metric=np.mean(metric_val, axis=0))




def train_batch_darcy(model, loss_func, data, optimizer, lr_scheduler, device, grad_clip=0.99):
    optimizer.zero_grad()
    a, x, edge = data["coeff"].to(device), data["node"].to(
        device), data["edge"].to(device)
    pos, grid = data['pos'].to(device), data['grid'].to(device)
    u, gradu = data["target"].to(device), data["target_grad"].to(device)
    pos_all = data['pos_all'].to(device)
    if config['model']['model_type'] in ['lite', 'ns']:
        pos = pos_all


    # print('in train-----------------------------------------------------------------------------------')
    # print('x, edge, pos, grid:', x.shape, edge.shape, pos.shape, grid.shape)
    # print('a:', a.shape)


    # pos is for attention, grid is the finest grid
    out_ = model(x, edge, pos=pos, grid=grid)
    if isinstance(out_, dict):
        out = out_['preds']
    elif isinstance(out_, tuple):
        out = out_[0]

    # print('out:', out.shape)
    if out.ndim == 4:
        
        u_pred, pred_grad, target = out[..., 0], out[..., 1:], u[..., 0]
        # print('u_pred, target, pred_grad, gradu,a:',u_pred.shape, target.shape, pred_grad.shape, gradu.shape,a.shape )
        loss, reg, _, _ = loss_func(u_pred, target, pred_grad, gradu, K=a)
    elif out.ndim == 3:
        u_pred, u = out[..., 0], u[..., 0]
        loss, reg, _, _ = loss_func(u_pred, u, targets_prime=gradu, K=a)
    loss = loss + reg
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    try:
        up_pred = out[..., 1:]
    except:
        up_pred = u_pred

    return (loss.item(), reg.item()), u_pred, up_pred


def validate_epoch_darcy(model, metric_func, valid_loader, device):
    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x, edge = data["node"].to(device), data["edge"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)
            out_ = model(x, edge, pos=pos, grid=grid)
            if isinstance(out_, dict):
                out = out_['preds']
            elif isinstance(out_, tuple):
                out = out_[0]
            u_pred = out[..., 0]
            target = data["target"].to(device)
            u = target[..., 0]
            _, _, metric, _ = metric_func(u_pred, u)
            try:
                metric_val.append(metric.item())
            except:
                metric_val.append(metric)

    return dict(metric=np.mean(metric_val, axis=0))


    model.eval()
    metric_val = []
    for _, data in enumerate(valid_loader):
        with torch.no_grad():
            x = data["node"].to(device)
            u = data["target"].to(device)
            pos, grid = data['pos'].to(device), data['grid'].to(device)

            out_ = model(x, None, pos=pos, grid=grid)
            u_pred = out_['preds']
            
            _, _, metric, _ = metric_func(u_pred, u)
        try:
            metric_val.append(metric.item())
        except:
            metric_val.append(metric)
        # metric_val.append(metric)

    return dict(metric=np.mean(metric_val, axis=0))


    def __init__(self, normalize: bool = False, is_masked: bool = False):
        super().__init__()
        self.normalize = normalize
        self.is_masked = is_masked

    def get_metric(self, pred, target):
        mse = self.MSE(pred, target)
        rmse = self.RMSE(pred, target)
        l2re = self.L2RE(pred, target)
        maxerror = self.MaxError(pred, target)
        nmse = self.NMSE(pred, target)
        mae = self.MAE(pred, target)

        return {'mse':mse, 
                'rmse':rmse, 
                'l2re':l2re, 
                'maxerror':maxerror, 
                'nmse':nmse, 
                'mae':mae}


    def MSE(self, pred, target):
        """return mean square error

        pred: model output tensor of shape (bs, x1, ..., xd, t, v)
        target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
        """
        assert pred.shape == target.shape
        temp_shape = [0, len(pred.shape)-1]
        temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
        pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
        target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
        nb, nc = pred.shape[0], pred.shape[1]
        errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
        res = torch.mean(errors**2, dim=2)
        return res # (bs, v)(32, 3)


    def RMSE(self, pred, target):
        """return root mean square error

        pred: model output tensor of shape (bs, x1, ..., xd, t, v)
        target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
        """
        return torch.sqrt(self.MSE(pred, target)) # (bs, v)


    def L2RE(self, pred, target):
        """l2 relative error (nMSE in PDEBench)

        pred: model output tensor of shape (bs, x1, ..., xd, t, v)
        target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
        """
        assert pred.shape == target.shape
        temp_shape = [0, len(pred.shape)-1]
        temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
        pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
        target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
        nb, nc = pred.shape[0], pred.shape[1]
        errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
        res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
        return torch.sqrt(res) # (bs, v)

    def MaxError(self, pred, target):
        """return max error in a batch

        pred: model output tensor of shape (bs, x1, ..., xd, t, v)
        target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
        """
        errors = torch.abs(pred - target)
        nc = errors.shape[-1]
        res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
        return res # (v)

    def NMSE(self, pred, target):
        assert pred.shape == target.shape
        temp_shape = [0, len(pred.shape)-1]
        temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
        pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
        target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
        nb, nc = pred.shape[0], pred.shape[1]
        errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
        norm = pred.reshape([nb, nc, -1])
        res = torch.sum(errors**2, dim=2) / torch.sum(norm**2, dim=2)
        return res

    def MAE(self, pred, target):
        # mean absolute error
        assert pred.shape == target.shape
        temp_shape = [0, len(pred.shape)-1]
        temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
        pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
        target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
        nb, nc = pred.shape[0], pred.shape[1]
        errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
        return torch.mean(errors.abs(),dim=-1) # (bs, v)

    def print_res(self, res):
        for u, v in res.items():
            dim = len(v)
            if dim == 1:
                print(u, "{0:.6f}".format(v.item()))
            else:
                for i in range(dim):
                    if i == 0:
                        print(u, "\t{0:.6f}".format(v[i].item()), end='\t')
                    else:
                        print("{0:.6f}".format(v[i].item()), end='\t')
                print("")
        return

    def write_res(self, res, filename, tag, append=True):
        df = pd.DataFrame()
        # Iterate over the metrics in res
        for metric, values in res.items():
            dim = len(values)
            values = [x.item() for x in values]
            if "field" not in df.columns:
                if dim > 1:
                    df.insert(0, 'field', [tag+"_x"+str(k) for k in range(dim)]+[tag + "_mean"])
                else:
                    df.insert(0, 'field', [tag+"_x"+str(k) for k in range(dim)])
            if dim > 1:
                df[metric] = [*values]+ [np.mean(values)]
            else:
                df[metric] = [*values]
        
        if append:
            if os.path.exists(filename):
                df.to_csv(filename, mode='a', header=False, index=False)
            else:
                df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)