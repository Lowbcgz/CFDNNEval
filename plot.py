from utils import get_model, get_dataset, get_dataloader, get_min_max, setup_seed
from functools import reduce
from pathlib import Path
from torch import Tensor
from typing import Optional
import yaml
import argparse
import numpy as np
import os
import typing
import matplotlib.pyplot as plt
import torch

@typing.no_type_check
def plot_predictions(
    label: Tensor,
    pred: Tensor,
    out_dir: Path,
    message: str,
):
    assert all([isinstance(x, Tensor) for x in [label, pred]])
    assert label.shape == pred.shape, f"{label.shape}, {pred.shape}"

    label_dir = out_dir / "label"
    label_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(exist_ok=True, parents=True)
    error_dir = out_dir / "error"
    error_dir.mkdir(exist_ok=True, parents=True)

    pred_arr = pred.squeeze().cpu().detach().numpy()
    label_arr = label.squeeze().cpu().detach().numpy()

    # Plot and save images
    u_min = min(pred_arr.min(), label_arr.min())
    u_max = max(pred_arr.max(), label_arr.max())

    #ground truth
    fig, ax = plt.subplots(figsize = (4,4))
    plt.axis("off")
    ax.contourf(
        label_arr, vmin=u_min, vmax=u_max, levels=150, cmap='jet'
    )
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(label_dir / f"{message}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    #pred
    fig, ax = plt.subplots(figsize = (4,4))
    plt.axis("off")
    ax.contourf(
        pred_arr, vmin=u_min, vmax=u_max, levels=150, cmap='jet'
    )
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(pred_dir / f"{message}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

    #error
    fig, ax = plt.subplots(figsize = (4,4))
    plt.axis("off")
    ax.contourf(
        np.abs(pred_arr - label_arr), vmin=u_min, vmax=u_max, levels=150
    )
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(error_dir / f"{message}.png", bbox_inches="tight", pad_inches=0)
    plt.close()

def test_plot(test_loader, model, device, fig_dir, error_dir, args):
    model.eval()

    gt_list = []
    pred_list = []

    with torch.no_grad():
        (channel_min, channel_max) = args["channel_min_max"] 
        channel_min, channel_max = channel_min.to(device), channel_max.to(device)
        prev_case_id = 0
        if_init = True
        #record frames
        for x, y, mask, case_params, grid, case_id in test_loader:
            case_id = case_id.item()
            if prev_case_id != case_id:
                break
            
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            case_params = case_params.to(device)
            grid = grid.to(device)

            y = y * mask
            if args["flow_name"] not in ["Darcy", "TGV"]:
                x = (x - channel_min)/(channel_max-channel_min) # normalization
                y = (y - channel_min)/(channel_max-channel_min)
            gt_list.append(y)
            
            if if_init:
                if_init = False
            else:
                x = pred.detach().clone()
        
            pred = model(x, case_params, mask, grid).reshape(-1, 64, 64, 3)
            # pred = pred*(channel_max - channel_min) + channel_min
            # pred = pred * mask
            pred_list.append(pred)
            
        total_frames = len(pred_list)
        print(f'total frames is {total_frames}')
        
        for i in range(len(pred_list)):
            # unnormalize
            pred_list[i] = pred_list[i]*(channel_max - channel_min) + channel_min
            gt_list[i] = gt_list[i]*(channel_max - channel_min) + channel_min
            
        #plot, only for regular grid
        assert len(pred_list) == len(gt_list)
        if "ir" not in args["dataset"]["case_name"]:
            cnt = 0
            time_list = ['0', '0.25T', '0.5T', '0.75T', 'T']
            for i in range(0, len(pred_list), len(pred_list) // 4):
                pred = pred_list[i]
                gt = gt_list[i]
                pred = pred.reshape(-1, 64, 64, 3)
                gt = gt.reshape(-1, 64, 64, 3)
                time = time_list[cnt]
                if args["flow_name"] in ['NSCH']:
                    U_pred = torch.sqrt(pred[..., 1]**2 + pred[..., 2]**2)
                    U_gt   = torch.sqrt(gt[..., 1]**2 + gt[..., 2]**2)
                    plot_predictions(label = U_gt, pred = U_pred, out_dir=Path(fig_dir), message=f'velocity_at_' + time)
                    plot_predictions(label = gt[...,0], pred = pred[...,0], out_dir=Path(fig_dir), message=f'f_at_' + time)
                else:
                    U_pred = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
                    U_gt   = torch.sqrt(gt[..., 0]**2 + gt[..., 1]**2)
                    plot_predictions(label = U_gt, pred = U_pred, out_dir=Path(fig_dir), message=f'velocity_at_' + time)
                    if args["flow_name"] not in ['tube']:
                        plot_predictions(label = gt[...,2], pred = pred[...,2], out_dir=Path(fig_dir), message=f'pressure_at_' + time)
                cnt += 1
            
            if cnt != 5:
                pred = pred_list[-1]
                gt = gt_list[-1]
                pred = pred.reshape(-1, 64, 64, 3)
                gt = gt.reshape(-1, 64, 64, 3)
                time = time_list[-1]
                if args["flow_name"] in ['NSCH']:
                    U_pred = torch.sqrt(pred[..., 1]**2 + pred[..., 2]**2)
                    U_gt   = torch.sqrt(gt[..., 1]**2 + gt[..., 2]**2)
                    plot_predictions(label = U_gt, pred = U_pred, out_dir=Path(fig_dir), message=f'velocity_at_' + time)
                    plot_predictions(label = gt[...,0], pred = pred[...,0], out_dir=Path(fig_dir), message=f'f_at_' + time)
                else:
                    U_pred = torch.sqrt(pred[..., 0]**2 + pred[..., 1]**2)
                    U_gt   = torch.sqrt(gt[..., 0]**2 + gt[..., 1]**2)
                    plot_predictions(label = U_gt, pred = U_pred, out_dir=Path(fig_dir), message=f'velocity_at_' + time)
                    if args["flow_name"] not in ['tube']:
                        plot_predictions(label = gt[...,2], pred = pred[...,2], out_dir=Path(fig_dir), message=f'pressure_at_' + time)
        
        #get error
        mse = []
        nmse = []
        max_error = []

        for i in range(len(pred_list)):
            pred = pred_list[i]
            gt = gt_list[i]

            nc = pred.shape[-1]
            pred = pred.reshape(-1, nc)
            gt = gt.reshape(-1, nc)

            error = pred - gt
            error = error*(channel_max - channel_min) + channel_min
            mse.append(torch.mean(error ** 2, axis=0).cpu().numpy())
            nmse.append((torch.mean(error ** 2, axis=0) / torch.mean(gt ** 2, axis=0)).cpu().numpy())
            max_error.append(np.max(np.abs(error.cpu().numpy()), axis=0))
            print(f"frame {i} mse: {np.mean(mse)} nmse: {np.mean(nmse)} max_error: {np.mean(max_error)}")

        np.save(os.path.join(error_dir, 'mse.npy'), np.array(mse))
        np.save(os.path.join(error_dir, 'nmse.npy'), np.array(nmse))
        np.save(os.path.join(error_dir, 'max_error.npy'), np.array(max_error))
                      
def main(args):
    #init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_dir = os.path.join(args["saved_dir"], os.path.join(args["model_name"], args["flow_name"] + '_' + args['dataset']['case_name']))

    fig_dir = os.path.join(args['output_dir'], 'fig')
    fig_dir = os.path.join(fig_dir, os.path.join(args["model_name"],args["flow_name"] + '_' + args['dataset']['case_name']))

    error_dir = os.path.join(args['output_dir'], 'error')
    error_dir = os.path.join(error_dir, os.path.join(args["model_name"],args["flow_name"] + '_' + args['dataset']['case_name']))

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    dataset_args = args["dataset"]

    saved_model_name = (args["model_name"] + 
                        f"_lr{args['optimizer']['lr']}" +
                        f"_bs{args['dataloader']['train_batch_size']}" +
                        args["flow_name"] + 
                        dataset_args['case_name']
                        )

    saved_path = os.path.join(saved_dir, saved_model_name)
    
    # data get dataloader
    train_data, val_data, test_data, test_ms_data = get_dataset(args)
    train_loader, val_loader, test_loader, test_ms_loader = get_dataloader(train_data, val_data, test_data, test_ms_data, args)

    # set some train args
    input, output, _, case_params, grid, _, = next(iter(val_loader))
    print("input tensor shape: ", input.shape[1:])
    print("output tensor shape: ", output.shape[1:] if val_loader.dataset.multi_step_size==1 else output.shape[2:])
    spatial_dim = grid.shape[-1]
    n_case_params = case_params.shape[-1]
    args["model"]["num_points"] = reduce(lambda x,y: x*y, grid.shape[1:-1])  # get num_points, especially of irregular geometry(point clouds)

    # get min_max per channel of train-set on the fly for normalization.
    channel_min, channel_max = get_min_max(train_loader)
    print("use min_max normalization with min=", channel_min.tolist(), ", max=", channel_max.tolist())
    print(channel_max, channel_min)
    args["channel_min_max"] = (channel_min, channel_max)

    #model
    model = get_model(spatial_dim, n_case_params, args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args['model_name']} has {num_params} parameters")

    #load checkpoint
    checkpoint = torch.load(saved_path + "-best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    #test and plot
    print('start ploting...')
    test_plot(test_loader, model, device, fig_dir, error_dir, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("-c", "--case_name", type=str, default="", help="For the case, if no value is entered, the yaml file shall prevail")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    if len(cmd_args.case_name) > 0:
        args['dataset']['case_name'] = cmd_args.case_name
    args['dataset']['multi_step_size'] = 1

    setup_seed(args["seed"])
    print(args)
    main(args)