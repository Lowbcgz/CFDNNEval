import numpy as np
import torch
from torch.nn.modules.loss import _WeightedLoss


import matplotlib.pyplot as plt

__all__ = ["UnitGaussianNormalizer", "WeightedL2Loss", "WeightedL2Loss2d", "WeightedL2Loss3d"]

class UnitGaussianNormalizer:
    def __init__(self, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()
        '''
        modified from utils3.py in 
        https://github.com/zongyi-li/fourier_neural_operator
        Changes:
            - .to() has a return to polymorph the torch behavior
            - naming convention changed to sklearn scalers 
        '''
        self.eps = eps

    def fit_transform(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)
        return (x - self.mean) / (self.std + self.eps)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return (x * (self.std + self.eps)) + self.mean

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.float().to(device)
            self.std = self.std.float().to(device)
        else:
            self.mean = torch.from_numpy(self.mean).float().to(device)
            self.std = torch.from_numpy(self.std).float().to(device)
        return self

    def cuda(self, device=None):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cuda(device)
        self.std = self.std.float().cuda(device)
        return self

    def cpu(self):
        assert torch.is_tensor(self.mean)
        self.mean = self.mean.float().cpu()
        self.std = self.std.float().cpu()
        return self

class WeightedL2Loss(_WeightedLoss):
    def __init__(self,
                 dilation=2,  # central diff
                 regularizer=False,
                 h=1/512,  # mesh size
                 beta=1.0,  # L2 u
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 metric_reduction='L1',
                 periodic=False,
                 return_norm=True,
                 orthogonal_reg=False,
                 orthogonal_mode='global',
                 delta=1e-4,
                 noise=0.0,
                 debug=False
                 ):
        super(WeightedL2Loss, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma*h  # H^1
        self.alpha = alpha*h  # H^1
        self.delta = delta*h  # orthongalizer
        self.eps = 1e-8
        # TODO: implement different bc types (Neumann)
        self.periodic = periodic
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.orthogonal_reg = orthogonal_reg
        self.orthogonal_mode = orthogonal_mode
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def central_diff(self, x: torch.Tensor, h=None):
        h = self.h if h is None else h
        d = self.dilation  # central diff dilation
        grad = (x[:, d:] - x[:, :-d])/d
        # grad = F.pad(grad, (1,1), 'constant', 0.)  # pad is slow
        return grad/h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                preds_latent: list = [], K=None):
        r'''
        all inputs are assumed to have shape (N, L)
        grad has shape (N, L) in 1d, and (N, L, 2) in 2D
        relative error in 
        \beta*\|N(u) - u\|^2 + \alpha*\| N(Du) - Du\|^2 + \gamma*\|D N(u) - Du\|^2
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        on uniform mesh, h can be set to 1
        preds_latent: (N, L, E)
        '''
        batch_size = targets.size(0)

        h = self.h
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        target_norm = h*targets.pow(2).sum(dim=1)

        if targets_prime is not None:
            targets_prime_norm = h*targets_prime.pow(2).sum(dim=1)
        else:
            targets_prime_norm = 1

        loss = self.beta * (h*(preds - targets).pow(2)).sum(dim=1)/target_norm

        if preds_prime is not None and self.alpha > 0:
            grad_diff = h*(preds_prime - K*targets_prime).pow(2)
            loss_prime = self.alpha * grad_diff.sum(dim=1)/targets_prime_norm
            loss += loss_prime

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch
            metric = loss.sqrt().max().item()

        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        if self.regularizer and self.gamma > 0 and targets_prime is not None:
            preds_diff = self.central_diff(preds)
            s = self.dilation // 2
            regularizer = self.gamma*h*(targets_prime[:, s:-s]
                                        - preds_diff).pow(2).sum(dim=1)/targets_prime_norm

            regularizer = regularizer.sqrt().mean() if self.return_norm else regularizer.mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)

        if self.orthogonal_reg > 0 and preds_latent:
            ortho = []
            for y_lat in preds_latent:
                if self.orthogonal_mode in ['local', 'fourier']:
                    pred_mm = torch.matmul(
                        y_lat, y_lat.transpose(-2, -1))
                elif self.orthogonal_mode in ['global', 'galerkin', 'linear']:
                    pred_mm = torch.matmul(
                        y_lat.transpose(-2, -1), y_lat)

                with torch.no_grad():
                    mat_dim = pred_mm.size(-1)
                    if self.orthogonal_mode in ['local', 'fourier']:
                        tr = (y_lat**2).sum(dim=-1)
                    elif self.orthogonal_mode in ['global', 'galerkin', 'linear']:
                        tr = (y_lat**2).sum(dim=-2)
                    assert tr.size(-1) == mat_dim
                    diag = [torch.diag(tr[i, :]) for i in range(batch_size)]
                    diag = torch.stack(diag, dim=0)
                ortho.append(
                    self.delta * ((pred_mm - diag)**2).mean(dim=(-1, -2)))
            orthogonalizer = torch.stack(ortho, dim=-1)
            orthogonalizer = orthogonalizer.sqrt().mean(
            ) if self.return_norm else orthogonalizer.mean()
        else:
            orthogonalizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)

        return loss, regularizer, orthogonalizer, metric


class WeightedL2Loss2d(_WeightedLoss):
    def __init__(self, flow_name, 
                 dim=2,
                 dilation=2,  # central diff
                 regularizer=False,
                 h=1/421,  # mesh size
                 beta=1.0,  # L2 u
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 delta=0.0,  #
                 metric_reduction='L1',
                 return_norm=True,
                 noise=0.0,
                 eps=1e-10,
                 debug=False
                 ):
        super(WeightedL2Loss2d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.dim = dim
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.alpha = alpha  # H^1
        self.delta = delta*h**dim  # orthogonalizer
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.debug = debug
        self.flow_name = flow_name

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    def central_diff(self, u: torch.Tensor, h=None):
        '''
        u: function defined on a grid (bsz, n, n)
        out: gradient (N, n-2, n-2, 2)
        '''
        bsz = u.size(0)
        h = self.h if h is None else h
        d = self.dilation  # central diff dilation
        s = d // 2  # central diff stride
        if self.dim > 2:
            raise NotImplementedError(
                "Not implemented: dim > 2 not implemented")

        grad_x = (u[:, d:, s:-s] - u[:, :-d, s:-s])/d
        grad_y = (u[:, s:-s, d:] - u[:, s:-s, :-d])/d
        grad = torch.stack([grad_x, grad_y], dim=-2)
        # if self.flow_name == 'darcy':
        #     grad = torch.stack([grad_x, grad_y], dim=-2)
        # else:
        #     grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad/h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                weights=None, K=None):
        r'''
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        targets_prime: (N, n, n, 1)
        K: (N, n, n, 1)
        beta * \|N(u) - u\|^2 + \alpha * \| N(Du) - Du\|^2 + \gamma * \|D N(u) - Du\|^2
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        '''
        batch_size = targets.size(0) # for debug only

        h = self.h if weights is None else weights
        d = self.dim
        K = torch.tensor(1) if K is None else K
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        target_norm = targets.pow(2).mean(dim=(1, 2)) + self.eps

        if targets_prime is not None:
            targets_prime_norm = d * \
                (K*targets_prime.pow(2)).mean(dim=(1, 2, 3)) + self.eps
        else:
            targets_prime_norm = 1
 
        # mse  归一化？？？
        loss = self.beta*((preds - targets).pow(2)
                          ).mean(dim=(1, 2))/target_norm

        # +导数的平方误差  false
        if preds_prime is not None and self.alpha > 0:
            grad_diff = (K*(preds_prime - targets_prime)).pow(2)
            loss_prime = self.alpha * \
                grad_diff.mean(dim=(1, 2, 3))/targets_prime_norm
            loss += loss_prime

        # 其他指标
        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch
            metric = loss.sqrt().max().item()

        # root -> rmse
        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        # 正则化项
        if self.regularizer and targets_prime is not None:
            preds_diff = self.central_diff(preds)
            s = self.dilation // 2
            targets_prime = targets_prime[:, s:-s, s:-s, :].contiguous()

            if K.ndim > 1:
                K = K[:, s:-s, s:-s].contiguous()

            # print('in regularizer-----------------------------------------------------------------')
            # print('K, targets_prime, preds_diff', K.shape, targets_prime.shape, preds_diff.shape)
            regularizer = self.gamma * h * ((K * (targets_prime - preds_diff))
                                            .pow(2)).mean(dim=(1, 2, 3))/targets_prime_norm

            regularizer = regularizer.sqrt().mean() if self.return_norm else regularizer.mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)
        norms = dict(L2=target_norm,
                     H1=targets_prime_norm)

        return loss, regularizer, metric, norms

class WeightedL2Loss3d(_WeightedLoss):
    def __init__(self,
                 dim=3,
                 dilation=2,  # central diff
                 regularizer=False,  # True
                 h=1/421,  # mesh size
                 beta=1.0,  # L2 u
                 gamma=1e-1,  # \|D(N(u)) - Du\|,
                 alpha=0.0,  # L2 \|N(Du) - Du\|,
                 delta=0.0,  #
                 metric_reduction='L1',
                 return_norm=True,
                 noise=0.0,
                 eps=1e-10,
                 debug=False
                 ):
        super(WeightedL2Loss3d, self).__init__()
        self.noise = noise
        self.regularizer = regularizer
        assert dilation % 2 == 0
        self.dilation = dilation
        self.dim = dim
        self.h = h
        self.beta = beta  # L2
        self.gamma = gamma  # H^1
        self.alpha = alpha  # H^1
        self.delta = delta*h**dim  # orthogonalizer
        self.eps = eps
        self.metric_reduction = metric_reduction
        self.return_norm = return_norm
        self.debug = debug

    @staticmethod
    def _noise(targets: torch.Tensor, n_targets: int, noise=0.0):
        assert 0 <= noise <= 0.2
        with torch.no_grad():
            targets = targets * (1.0 + noise*torch.rand_like(targets))
        return targets

    
    def central_diff(self, u: torch.Tensor, h=None):
        '''
        u: function defined on a grid (bsz, n, n, n)
        out: gradient (N, n-2, n-2, n-2, 3)
        '''
        bsz = u.size(0)
        h = self.h if h is None else h
        d = self.dilation  # central diff dilation
        s = d // 2  # central diff stride
        if self.dim > 3:
            raise NotImplementedError(
                "Not implemented: dim > 3 not implemented")

        # grad_x = (u[:, d:, s:-s] - u[:, :-d, s:-s])/d
        # grad_y = (u[:, s:-s, d:] - u[:, s:-s, :-d])/d
        grad_x = (u[:, d:, s:-s, s:-s] - u[:, :-d, s:-s, s:-s])/d  # (N, S_x, S_y, t)
        grad_y = (u[:, s:-s, d:, s:-s] - u[:, s:-s, :-d, s:-s])/d  # (N, S_x, S_y, t)
        grad_z = (u[:, s:-s, s:-s, d:] - u[:, s:-s, s:-s, :-d])/d  # (N, S_x, S_y, t)
        grad = torch.stack([grad_x, grad_y, grad_z], dim=-2)
        return grad/h

    def forward(self, preds, targets,
                preds_prime=None, targets_prime=None,
                weights=None, K=None):
        r'''
        preds: (N, n, n, n, 1)
        targets: (N, n, n, n, 1)
        targets_prime: (N, n, n, n, 3)
        K: (N, n, n, n, 1)   or   1
        beta * \|N(u) - u\|^2 + \alpha * \| N(Du) - Du\|^2 + \gamma * \|D N(u) - Du\|^2
        weights has the same shape with preds on nonuniform mesh
        the norm and the error norm all uses mean instead of sum to cancel out the factor
        '''
        batch_size = targets.size(0) # for debug only

        h = self.h if weights is None else weights
        d = self.dim
        K = torch.tensor(1) if K is None else K
        if self.noise > 0:
            targets = self._noise(targets, targets.size(-1), self.noise)

        # print('in loss-----------------------------------------------------------------------')
        # print('pred, target:', preds.shape, targets.shape)
        target_norm = targets.pow(2).mean(dim=(1, 2, 3)) + self.eps  # 1

        if targets_prime is not None:
            targets_prime_norm = d * \
                (K*targets_prime.pow(2)).mean(dim=(1, 2, 3, 4)) + self.eps  # 1
        else:
            targets_prime_norm = 1

        loss = self.beta*((preds - targets).pow(2)
                          ).mean(dim=(1, 2, 3))/target_norm  # 1

        # false
        if preds_prime is not None and self.alpha > 0:  
            grad_diff = (K*(preds_prime - targets_prime)).pow(2)
            loss_prime = self.alpha * \
                grad_diff.mean(dim=(1, 2, 3))/targets_prime_norm
            loss += loss_prime

        if self.metric_reduction == 'L2':
            metric = loss.mean().sqrt().item()
        # true  L1
        elif self.metric_reduction == 'L1':  # Li et al paper: first norm then average    
            metric = loss.sqrt().mean().item()
        elif self.metric_reduction == 'Linf':  # sup norm in a batch
            metric = loss.sqrt().max().item()

        loss = loss.sqrt().mean() if self.return_norm else loss.mean()

        # true  +正则化
        if self.regularizer and targets_prime is not None:  

            preds_diff = self.central_diff(preds)  # [bs, n, n, n, 3]
            s = self.dilation // 2
            targets_prime = targets_prime[:, s:-s, s:-s, s:-s,:].contiguous()

            if K.ndim > 1:
                K = K[:, s:-s, s:-s].contiguous()

            # print('in loss---------------------------------------------------')
            # print('K, targets_prime, preds_diff', K.shape,  targets_prime.shape, preds_diff.shape, 
                #   targets_prime_norm.shape)
            regularizer = self.gamma * h * ((K * (targets_prime - preds_diff))
                                            .pow(2)).mean(dim=(1, 2, 3, 4))/targets_prime_norm

            regularizer = regularizer.sqrt().mean() if self.return_norm else regularizer.mean()

        else:
            regularizer = torch.tensor(
                [0.0], requires_grad=True, device=preds.device)
        norms = dict(L2=target_norm,
                     H1=targets_prime_norm)

        return loss, regularizer, metric, norms

