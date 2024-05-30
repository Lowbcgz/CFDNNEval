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

    pred_arr = pred[0].cpu().detach().numpy()
    label_arr = label[0].cpu().detach().numpy()

    # Plot and save images
    u_min = min(pred_arr.min(), label_arr.min())
    u_max = max(pred_arr.max(), label_arr.max())

    #ground truth
    plt.axis("off")
    plt.imshow(
        label_arr, vmin=u_min, vmax=u_max, cmap="coolwarm"
    )
    plt.savefig(label_dir / f"{message}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    #pred
    plt.axis("off")
    plt.imshow(
        pred_arr, vmin=u_min, vmax=u_max, cmap="coolwarm"
    )
    plt.savefig(pred_dir / f"{message}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    #error
    plt.axis("off")
    plt.imshow(
        np.abs(pred_arr - label_arr), vmin=u_min, vmax=u_max, cmap="coolwarm"
    )
    plt.savefig(error_dir / f"{message}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

def test_plot(test_loader, model, device, output_dir, args):
    model.eval()

    x_list = []
    gt_list = []
    mask_list = []
    case_params_list = []
    grid_list = []
    pred_list = []

    with torch.no_grad():
        (channel_min, channel_max) = args["channel_min_max"] 
        channel_min, channel_max = channel_min.to(device), channel_max.to(device)
        prev_case_id = 0

        for x, y, mask, case_params, grid, case_id in test_loader:
            case_id = case_id.item()
            
            if args["flow_name"] not in ["Darcy", "TGV"]:
                x, y = x.to(device), y.to(device)
                x = (x - channel_min)/(channel_max-channel_min) # normalization
                y = (y - channel_min)/(channel_max-channel_min)
            x_list.append(x)
            gt_list.append(y)
            mask_list.append(mask)
            case_params_list.append(case_params)
            grid_list.append(grid)

            if prev_case_id != case_id:
                break
        
        total_frames = len(x_list)
        print(f'total frames is {total_frames}')
        
        for i in range(total_frames):
            if i == 0:
                x = x_list[0].to(device)
            else:
                x = pred.detach().clone()

            case_params = case_params_list[i].to(device)
            mask = mask_list[i].to(device)
            grid = grid_list[i].to(device)
            pred = model(x, case_params, mask, grid)

            pred_list.append(pred)
        
        assert len(pred_list) == len(gt_list)
        cnt = 0
        time_list = ['0', '0.25T', '0.5T', '0.75T', 'T']
        for i in range(0, len(pred_list), len(pred_list) // 4):
            pred = pred_list[i]
            gt = gt_list[i]
            time = time_list[cnt]
            for j in range(pred.shape[-1]):
                plot_predictions(label = gt[..., j], pred = pred[..., j], out_dir=Path(output_dir), message=f'variable{j}_at_' + time)
            cnt += 1
        
        if cnt != 5:
            pred = pred_list[-1]
            gt = gt_list[-1]
            time = time_list[-1]
            for j in range(pred.shape[-1]):
                plot_predictions(label = gt[..., j], pred = pred[..., j], out_dir=Path(output_dir), message=f'variable{j}_at_' + time)
                        
def main(args):
    #init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_dir = os.path.join(args["saved_dir"], os.path.join(args["model_name"], args["flow_name"] + '_' + args['dataset']['case_name']))

    fig_dir = os.path.join(args['output_dir'], './fig')
    output_dir = os.path.join(fig_dir, os.path.join(args["model_name"],args["flow_name"] + '_' + args['dataset']['case_name']))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    test_plot(test_loader, model, device, output_dir, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--test", action='store_true', help='test mode')
    parser.add_argument("--continue_training", action='store_true', help='continue training')
    parser.add_argument("-c", "--case_name", type=str, default="", help="For the case, if no value is entered, the yaml file shall prevail")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    args['if_training'] = not cmd_args.test
    args['continue_training'] = cmd_args.continue_training
    if len(cmd_args.case_name) > 0:
        args['dataset']['case_name'] = cmd_args.case_name

    setup_seed(args["seed"])
    print(args)
    main(args)