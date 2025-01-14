import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from utils import setup_seed, get_model, get_model_name, get_test_dataset


def main(args):
    # setup random seed
    setup_seed(args["seed"])

    # get default model path if it is not specified in config file or command line args
    if args["model_path"] == "None":
        model_name = get_model_name(args)
        default_model_path = os.path.join(args["saved_dir"], 
                                          args["flow_name"],
                                          args["dataset"]["case_name"],
                                          f"{model_name}-best.pt")
        args["model_path"] = default_model_path

    # check existence of model path
    assert os.path.exists(args["model_path"]), f"No checkpoint is found at {args['model_path']}."

    # get test data
    test_data = get_test_dataset(args)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)

    # load model from checkpoint
    checkpoint = torch.load(args["model_path"])
    model = get_model(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of model parameters to train:", total_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Load model from {args['model_path']}")
    print(f"Best epoch: {checkpoint['epoch']}")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # load channel min-max value
    if "channel_min_max" in checkpoint.keys():
        args["channel_min_max"] = checkpoint["channel_min_max"]
    else:
        args["channel_min_max"] = None
        assert cmd_args.denormalize is False

    # inference loop
    model.eval()
    prev_case_id = -1
    pred_list = []
    gt_list = []

    if args["channel_min_max"] is not None:
        channel_min, channel_max = args["channel_min_max"]
        channel_min, channel_max = channel_min.to(device), channel_max.to(device)

    for x, y, mask, case_params, grid, case_id in test_loader:
        if prev_case_id != case_id and prev_case_id != -1:
            break

        # prepare model input
        if prev_case_id != case_id:
            x = x.to(device)
            if args["channel_min_max"]:
                x = (x - channel_min) / (channel_max - channel_min)
        else:
            x = pred_list[-1].to(device)
        
        y = y.to(device)

        if args["channel_min_max"] and not cmd_args.denormalize:
            y = (y - channel_min) / (channel_max - channel_min)

        mask = mask.to(device)
        case_params = case_params.to(device)
        grid = grid.to(device)
        y = y * mask

        # inference
        with torch.no_grad():
            pred = model(x, case_params, mask, grid)
        pred_list.append(pred.cpu())

        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            y = y[..., args["model"]["var_id"]].unsqueeze(-1)
        gt_list.append(y.cpu())

        prev_case_id = case_id.item()

    print(len(pred_list), pred_list[0].shape)
    print(len(gt_list), gt_list[0].shape)
    
    # save inference result
    output_dir = os.path.join(cmd_args.result_dir, 
                              args["model_name"], 
                              f"{args['flow_name']}_{args['dataset']['case_name']}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
        torch.save(pred_list, os.path.join(output_dir, f"pred_x{args['model']['var_id']}_list.pt"))
        torch.save(gt_list, os.path.join(output_dir, f"gt_x{args['model']['var_id']}_list.pt"))
    else:
        torch.save(pred_list, os.path.join(output_dir, "pred_list.pt"))
        torch.save(gt_list, os.path.join(output_dir, "gt_list.pt"))
        

if __name__ == "__main__":
    # specific device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # can be accessed globally

    # parse args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file.")
    parser.add_argument("--case_name", type=str, help="Case name.")
    parser.add_argument("--model_path", type=str, help="Checkpoint path to test.")
    parser.add_argument("--denormalize", action="store_true", help="Compute metrics using denormalized output.")
    parser.add_argument("--result_dir", type=str, default="result")
    cmd_args = parser.parse_args()

    # read default args from config file
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)

    # update args using command args
    if cmd_args.model_path:
        args["model_path"] = cmd_args.model_path
    if cmd_args.case_name:
        args["dataset"]["case_name"] = cmd_args.case_name
    print(args)

    main(args)