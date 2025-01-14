import argparse
import h5py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from timeit import default_timer
from tqdm import tqdm

import metrics
from utils import setup_seed, get_test_dataset, get_model, get_model_name, append_results


METRICS = ['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE']


def test_loop(test_loader, model, args, metric_names=METRICS, test_type="accumulate"):
    assert test_type in ["frame", "accumulate"]

    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    cost_time = []

    model.eval()
    prev_case_id = -1
    preds = torch.Tensor().to(device)
    targets = torch.Tensor().to(device)

    if args["channel_min_max"] is not None:
        channel_min, channel_max = args["channel_min_max"]
        channel_min, channel_max = channel_min.to(device), channel_max.to(device)

    t1 = default_timer()
    for x, y, mask, case_params, grid, case_id in tqdm(test_loader):
        if prev_case_id != case_id:
            if prev_case_id != -1: # compute metric here
                t2 = default_timer()
                cost_time.append(t2-t1)
                if cmd_args.denormalize: # denormalize
                    if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
                        preds = preds * (channel_max[args["model"]["var_id"]] - channel_min[args["model"]["var_id"]]) + channel_min[args["model"]["var_id"]]
                    else:
                        preds = preds * (channel_max - channel_min) + channel_min
                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    res_dict[name].append(metric_fn(preds, targets))
                t1 = default_timer()
            preds = torch.Tensor().to(device)
            targets = torch.Tensor().to(device)

        if test_type == "frame" or prev_case_id != case_id:
            x = x.to(device)
            if args["channel_min_max"] is not None: 
                # normalize input
                x = (x - channel_min) / (channel_max - channel_min)
        else:
            x = preds[:, -1]

        y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
        if args["channel_min_max"] is not None and not cmd_args.denormalize:
            y = (y - channel_min) / (channel_max - channel_min)
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        mask = mask.to(device) # mask [b, x1, ..., xd, 1]
        case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
        y = y * mask
        
        with torch.no_grad():
            pred = model(x, case_params, mask, grid) # [bs, h, w, c] (mpnn: [bs, h, w, 1])

        # update
        preds = torch.cat([preds, pred.unsqueeze(1)], dim=1) # [bs, t, h, w, c] (mpnn: [bs, t, h, w, 1])
        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            y = y[..., args["model"]["var_id"]].unsqueeze(-1) # [bs, t, h, w, 1]
        targets = torch.cat([targets, y.unsqueeze(1)], dim=1) # [bs, t, h, w, c] (mpnn: [bs, t, h, w, 1])

        prev_case_id = case_id
    
    # compute metrics for last case
    t2 = default_timer()
    cost_time.append(t2-t1)
    if cmd_args.denormalize: # denormalize
        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            preds = preds * (channel_max[args["model"]["var_id"]] - channel_min[args["model"]["var_id"]]) + channel_min[args["model"]["var_id"]]
        else:
            preds = preds * (channel_max - channel_min) + channel_min
    for name in metric_names:
        metric_fn = getattr(metrics, name)
        res_dict[name].append(metric_fn(preds, targets))
    
    # post process
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res

    print(f"Total test time: {sum(cost_time):.4f}s, average {sum(cost_time) / len(cost_time):.4f}s per case.")

    return res_dict


def main(args):
    setup_seed(args["seed"])

    # check mode
    assert not args["if_training"]

    # get default model path if it is not specified in config file or command line args
    if args["model_path"] == "None":
        model_name = get_model_name(args)
        default_model_path = os.path.join(args["saved_dir"], 
                                          args["flow_name"],
                                          args["dataset"]["case_name"],
                                          f"{model_name}-best.pt")
        args["model_path"] = default_model_path

    # check existence of model path
    assert os.path.exists(args["model_path"]), f"No checkpoint found at {args['model_path']}."

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

    # test
    print("Start testing.")
    res_dict = test_loop(test_loader, model, args, test_type=cmd_args.test_type)

    # print results
    for k in res_dict:
        print(f"{k}: {res_dict[k]}")

    # save results
    if cmd_args.save_result:
        result_path = os.path.join(cmd_args.result_root, f"{args['model_name']}.csv")

        # collect result data
        _, metric_value = next(iter(res_dict.items()))
        num_vars = metric_value.shape[0]

        if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
            result_data = {}
            result_data["Field"] = f"{args['flow_name']}_{args['dataset']['case_name']}_x{args['model']['var_id']}"
            for metric_name, metric_value in res_dict.items():
                result_data[metric_name] = metric_value.cpu().item()
            append_results(result_data, result_path)

        else:
            result_data_list = []
            for var_id in range(num_vars):
                result_data = {}
                result_data["Field"] = f"{args['flow_name']}_{args['dataset']['case_name']}_x{var_id}"
                for metric_name, metric_value in res_dict.items():
                    result_data[metric_name] = metric_value[var_id].cpu().item()
                result_data_list.append(result_data)
            append_results(result_data_list, result_path)


if __name__ == "__main__":
    # specific device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # can be accessed globally

    # parse args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file.")
    parser.add_argument("-c", "--case_name", type=str, help="Case name.")
    parser.add_argument("--model_path", type=str, help="Checkpoint path to test.")
    parser.add_argument("--test_type", type=str, default="accumulate", help="Checkpoint path to test.")
    parser.add_argument("--denormalize", action="store_true", help="Compute metrics using denormalized output.")
    parser.add_argument("--save_result", action="store_true", help="Save results if set.")
    parser.add_argument("--result_root", type=str, default="result", help="Root path to save results (default: 'result').")
    cmd_args = parser.parse_args() # can be accessed globally

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