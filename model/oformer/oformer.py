import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from typing import Optional

from model.oformer.encoder_module import IrregSTEncoder2D
from model.oformer.decoder_module import IrregSTDecoder2D, IrregSTDecoder3D

from timeit import default_timer

class Oformer(nn.Module):
    """
    The overall network. It contains encoder and decoder.
    """
    def __init__(self, input_ch, output_ch, n_tolx, multi_step_size, dim=2) -> None:
        super(Oformer, self).__init__()
        self.encoder = IrregSTEncoder2D(
            input_channels=input_ch,    # vx, vy, prs, dns, pos_x, pos_y
            time_window=1,
            in_emb_dim=128,
            out_chanels=128,
            max_node_type=3,
            heads=1,
            depth=4,
            res=200,
            use_ln=True,
            emb_dropout=0.0,
        )
        IrregSTDecoder = IrregSTDecoder3D if dim==3 else IrregSTDecoder2D
        self.decoder = IrregSTDecoder(
            max_node_type=3,
            latent_channels=128,
            out_channels=output_ch,  # vx, vy, prs, dns
            res=200,
            scale=2,
            dropout=0.1
        )
        self.n_tolx = n_tolx
        self.multi_step_size = multi_step_size
        # total_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) +\
        #                 sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        # print(f'Total trainable parameters: {total_params}')
    
    def forward(self, x, case_params, node_type, grid, multi_step_size=None):
        if multi_step_size is None:
            multi_step_size = self.multi_step_size
        # print(x.shape, case_params.shape, node_type.shape, grid.shape)
        x = x.reshape(-1, 1, self.n_tolx, x.shape[-1])
        node_type = node_type.long().reshape(-1, multi_step_size, self.n_tolx, node_type.shape[-1])
        grid = grid.reshape(-1, self.n_tolx, grid.shape[-1])
        case_params = case_params.reshape(-1, 1, self.n_tolx, case_params.shape[-1])

        input_pos = prop_pos = grid
        fx   = torch.cat((x, case_params), dim=-1).detach()
        z    = self.encoder.forward(fx, node_type[:,0,...].detach(), input_pos)
        pred = self.decoder.forward(z, prop_pos, node_type[:,0,...].detach(), multi_step_size, input_pos)
        return pred
    
    def one_forward_step(self, x, case_params, mask, grid, y, loss_fn=None, args= None):
        info = {}
        pred = self(x, case_params, mask, grid)
        
        if loss_fn is not None:
            ## defined your specific loss calculations here
            y = y.reshape(-1, self.multi_step_size, self.n_tolx, y.shape[-1])
            loss = loss_fn(pred, y)
            return loss, pred, info
        else:
            #TODO: default loss_fn
            pass