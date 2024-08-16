import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import metrics
from timeit import default_timer
from functools import reduce

from utils import setup_seed, process_data
from visualize import *
from dataset import *
from utils import get_min_max

from libs_path import *
from libs import *
from libs.ns_lite import *
from dataset import *
from train import get_dataset, get_dataloader, get_model

def train_loop(model, train_loader, optimizer, scheduler, loss_fn, device, args, grad_clip=0.99):
    model.train()
    t1 = default_timer()
    train_loss = 0
    train_l_inf = 0
    step = 0
    # 每个batch传出一个loss
    for data in train_loader:  
        step += 1
        # batch_size = x.size(0)
        loss_total = 0
        reg_total = 0
        if args['dataset']['flow_name'] in ['cylinder', 'TGV']:
            pass
        else:
            data = process_data(args, data)

        x, edge = data["node"].to(
        device), data["edge"].to(device)
        pos_all = data['pos_all'].to(device)
        pos, grid = data['pos'].to(device), data['grid'].to(device)
        u, gradu = data["target"].to(device), data["target_grad"].to(device)
        mask = data['mask'].to(device)
        # u = u * mask
        # print('in train-----------------------------------------------------')
        # print(u.shape, mask.shape)
        
        if args['model']['model_type'] in ['lite', 'ns', '3D']:
            pos = pos_all

        steps = u.size(-1)  # 3steps, every step with u,v,p
        steps_long = int(steps / 3)
        if steps_long==0:
            steps_long = 1

        # if not args['if_norm']:
        #     (channel_min, channel_max) = args["channel_min_max"] 
        #     channel_min, channel_max = channel_min.to(device), channel_max.to(device)


        if args["training_type"] in ['autoregressive']:
            if getattr(train_loader.dataset,"multi_step_size", 1) ==1:
                #Model run one_step
                # if not args['if_norm']:
                #     x[..., :steps_long] = (x[..., :steps_long] - channel_min)/(channel_max-channel_min) # x: [bs, n, n, 3+p]
                #     u = (u - channel_min)/(channel_max-channel_min)  # u: [bs, n, n, 3*3]

                u = u * mask
                
                out_ = model(x, edge, pos=pos, grid=grid)  # in:[4, 64, 64, 2+p] --> out:[4, 64, 64, 2]
                if isinstance(out_, dict):
                    pred = out_['preds']
                elif isinstance(out_, tuple):
                    pred = out_[0]
                # print('in dataset---------------------------------------------------')
                # print('pred, u,targets_prime', pred.shape, u.shape,gradu.shape)
                loss_u, reg_u, _, _ = loss_fn(pred, u,
                                    targets_prime=gradu)
                loss_u = loss_u + reg_u
                loss_total += loss_u
                reg_total += reg_u.item()
                
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                _batch = pred.size(0)
                train_loss += loss_total.item()
                train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - u.reshape(_batch, -1)))))
            else:
                # norm
                # if not args['if_norm']:
                #     x[..., :steps_long] = (x[..., :steps_long] - channel_min)/(channel_max-channel_min) # x: [bs, n, n, 3+p]
                #     u = (u - channel_min.repeat(3))/(channel_max.repeat(3)-channel_min.repeat(3))  # u: [bs, n, n, 3*3]
                u = u * mask

                # Autoregressive loop
                preds=[]
                y = []
                loss_total = 0
                reg_total = 0
                for t in range(0, steps, steps_long):
                    out_ = model(x, edge, pos=pos, grid=grid)  # in:[4, 64, 64, 2+p] --> out:[4, 64, 64, 2]
                    if isinstance(out_, dict):
                        pred = out_['preds']
                    elif isinstance(out_, tuple):
                        pred = out_[0]
                    u_step = u[..., t:t+steps_long]
                    gradu_step = gradu[..., t:t+steps_long]
                    # 分通道计算loss并求和
                    for i in range(steps_long):  # steps_long=2:u, v      steps_long=3:u, v, p
                        loss_u, reg_u, _, _ = loss_fn(pred[..., i], u_step[..., i],
                                                targets_prime=gradu_step[..., i])
                        loss_u = loss_u + reg_u
                        loss_total += loss_u
                        reg_total += reg_u.item()
                
                    x = torch.cat((u_step, x[..., steps_long:]), dim=-1)
                    preds.append(pred)
                    y.append(u_step)

                preds=torch.stack(preds, dim=1)
                y=torch.stack(y, dim=1)
                _batch = preds.size(0)
                # loss = loss_fn(preds.reshape(_batch, -1), y.reshape(_batch, -1))
                optimizer.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()  # step batch

                train_loss += loss_total.item()
                train_l_inf = max(train_l_inf, torch.max((torch.abs(preds.reshape(_batch, -1) - y.reshape(_batch, -1)))))
    train_loss /= step      
    t2 = default_timer()
    return train_loss, train_l_inf, t2 - t1

def main(args):
    #init
    device = torch.device(args['device'] if torch.cuda.is_available() else "cpu")
    dataset_args = args["dataset"]

    saved_model_name = (args["model_name"] + 
                        f"_lr{args['optimizer']['lr']}" +
                        f"_model{args['model']['model_type']}" +
                        f"_bs{args['dataloader']['train_batch_size']}" +'_'+
                        dataset_args["flow_name"] + '_' +
                        dataset_args['case_name']
                        )
    print('saved_model_name:', saved_model_name)
    
    # data get dataloader
    train_data, val_data, test_data, test_ms_data = get_dataset(args)
    train_loader, val_loader, test_loader, test_ms_loader = get_dataloader(train_data, val_data, test_data, test_ms_data, args)


    # set some train args
    data = next(iter(val_loader))
    # if args['dataset']['flow_name'] in ['cylinder', 'cavity']:
    if args['dataset']['flow_name'] in ['cylinder', 'TGV']:
        pass
    else:
        data = process_data(args, data)


    input = data['node']
    output = data['target']
    grid = data['grid']
    # target_grad = data['target_grad']
    print("input tensor shape: ", input.shape[1:])
    print("output tensor shape: ", output.shape[1:] if val_loader.dataset.multi_step_size==1 else output.shape[2:])
    spatial_dim = grid.shape[-1]
    # print('grid', grid.shape)
    # print('spatial_dim', spatial_dim)
    # n_case_params = case_params.shape[-1]
    args["model"]["num_points"] = reduce(lambda x,y: x*y, grid.shape[1:-1])  # get num_points, especially of irregular geometry(point clouds)
    
    
    # add norm
    if not args['if_norm']:
        # if args['dataset']['flow_name'] in ['cylinder', 'cavity']:
        if args['dataset']['flow_name'] in ['cylinder', 'TGV']:
            channel_min, channel_max = get_min_max(train_loader, args) 
        else:
            channel_min, channel_max = get_min_max(train_loader, args, process_data) 

        args["channel_min_max"] = (channel_min, channel_max)
        print("use min_max normalization with min=", channel_min.tolist(), ", max=", channel_max.tolist())
        train_loader.dataset.apply_norm(channel_min, channel_max)
        val_loader.dataset.apply_norm(channel_min, channel_max)
        # test_loader.dataset.apply_norm(channel_min, channel_max)
        # if test_ms_data is not None:
            # test_ms_loader.dataset.apply_norm(channel_min, channel_max)


    # add original setup
    subsample_nodes=3,
    subsample_attn=6,
    n_grid = input.shape[1]
    fine_grid = (n_grid-1) * subsample_nodes[0] + 1  # 逆过来求一下原始网格数(原文这里为421的原始网格)
    n_grid_c = int(((fine_grid - 1)/subsample_attn[0]) + 1)
    parser.add_argument('--subsample-nodes', type=int, default=subsample_nodes, metavar='subsample',
                        help=f'input fine grid sampling from 421x421 (default: {subsample_nodes} i.e., {n_grid}x{n_grid} grid)')
    parser.add_argument('--subsample-attn', type=int, default=6, metavar='subsample_attn',
                        help=f'input coarse grid sampling from 421x421 (default: {subsample_attn} i.e., {n_grid_c}x{n_grid_c} grid)')
    if args['model']['model_type'] == 'darcy':
        # darcy流
        n_x, n_y = input.shape[1], input.shape[2]
        args['n_x'] = n_x
        args['n_y'] = n_y
        model_config = args['model']
        # 使用原始网格计算模型中的降采样、升采样大小
        downsample, upsample = PDEDarcyDataset.get_scaler_sizes(n_grid, n_grid_c)
        model_config = args['model']
        model_config['downscaler_size'] = downsample
        model_config['upscaler_size'] = upsample
        model_config['attn_norm'] = not model_config['attn_norm']  # True
        model_config['node_feats'] = int(model_config['node_feats'])
        if model_config['attention_type'] == 'fourier' or n_grid < 211:
            model_config['norm_eps'] = 1e-7
        elif model_config['attention_type'] == 'galerkin' and n_grid >= 211:
            model_config['norm_eps'] = 1e-5

    #model
    model = get_model(spatial_dim, None, args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args['model_name']} has {num_params} parameters")

    model.to(device) 
    model.train()

    # optimizer 
    if args['model']['attention_type'] in ['fourier', 'softmax']:
        lr = min(args['optimizer']['lr'], 5e-4)
    else:
        lr = args['optimizer']['lr']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler

    scheduler = OneCycleLR(optimizer, max_lr=lr, 
                            div_factor=1e4, 
                            final_div_factor=1e4,
                            pct_start=0.3,
                            steps_per_epoch=len(train_loader), 
                            epochs=args['epochs'])

    h = 1/n_grid
    if args['model']['model_type'] == '3D':
        loss_fn = WeightedL2Loss3d(regularizer=True, h=h, gamma=args['model']['gamma'])
    else:
        loss_fn = WeightedL2Loss2d(args['dataset']['flow_name'], regularizer=True, h=h, gamma=args['model']['gamma'])

     # train loop
    print("start training...")
    total_time = 0
    for epoch in range(0, 4):
    # 4个epoch
        train_loss, train_l_inf, time = train_loop(model, train_loader, optimizer, scheduler, loss_fn, device, args)
        scheduler.step() # step batch
        print(f"[Epoch {epoch}] train_loss: {train_loss}, train_l_inf: {train_l_inf}, time_spend: {time:.3f}")
        total_time += time
    print("avg_time : {0:.5f}".format(total_time / 4))
    avg_time = str(total_time / 4)
    with open('epoch_time.txt','a') as f:    #设置文件对象
        f.write(avg_time+'\n')
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--test", action='store_true', help='test mode')
    parser.add_argument("--norm", action='store_true', help='norm mode')
    parser.add_argument("--continue_training", action='store_true', help='continue training')
    parser.add_argument("-c", "--case_name", type=str, default="", help="For the case, if no value is entered, the yaml file shall prevail")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    
    args['if_training'] = not cmd_args.test 
    args['if_norm'] = not cmd_args.norm
    args['continue_training'] = cmd_args.continue_training
    if len(cmd_args.case_name) > 0:
        args['dataset']['case_name'] = cmd_args.case_name
   
    # setup_seed(args["seed"])
    print(args)
    main(args)

    # python train_one