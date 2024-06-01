from utils import get_model, get_dataset, get_dataloader, get_min_max, setup_seed
import metrics
from functools import reduce
import yaml
import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from plot import plot_predictions
from pathlib import Path
from dataset import *

def get_test_dataset_target(args, target_case_name):
    dataset_args = args["dataset"]
    if(args["flow_name"] == "tube"):
        test_data = TubeDataset(filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
    elif args["flow_name"] == "cavity":
        test_data = CavityDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
    elif args["flow_name"] == "cylinder":
        test_data = CylinderDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        
    elif args["flow_name"] == "ircylinder":
        test_data = IRCylinderDataset(
                                filename='cylinder_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
    elif args["flow_name"] == "NSCH":
        test_data = NSCHDataset(
                                filename='test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
    elif args["flow_name"] == "TGV":
        test_data = TGVDataset(
                                filename='test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
    elif args['flow_name'] == 'Darcy':
        raise Exception("cases in 'Darcy is not compatible yet")
        
    elif args["flow_name"] == "hills":
        test_data = HillsDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
    elif args["flow_name"] == "irhills":
        test_data = IRHillsDataset(
                                filename=args['flow_name'][2:] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=target_case_name,
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        
    else:
        raise ValueError("Invalid flow name.")

    print("#Test data: ", len(test_data))

    return test_data

def test_loop(test_loader, model, device, output_dir, args, target_case_name, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], test_type = 'frames'):
    model.eval()
    step = 0
    case_count = 0
    assert getattr(test_loader.dataset,"multi_step_size", 1) == 1
    if test_type == 'frames':
        print('consider the result between frames')
    elif test_type == 'accumulate':
        print('consider the accumulate result')
    else:
        raise Exception(f"test_type {test_type} is not support")
    
    res_dict = {"cw_res":{},  # channel-wise
                "sw_res":{},}  # sample-wise
    for name in metric_names:
        res_dict["cw_res"][name] = []
        res_dict["sw_res"][name] = []

    prev_case_id = -1
    preds = []
    gts = []
    with torch.no_grad():
        for x, y, mask, case_params, grid, case_id in test_loader:
            case_id = case_id.item()
            if prev_case_id != case_id:
                prev_case_id = -1
                step = 0
                case_count = case_count + 1
            
            step += 1
            x = x.to(device)
            y = y.to(device) # y: target tensor  [b, x1, ..., xd, v] if mutli_step_size ==1 else [b, multi_step_size, x1, ..., xd, v]
            grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims] 
            mask = mask.to(device) # mask [b, x1, ..., xd, 1] if mutli_step_size ==1 else [b, multi_step_size, x1, ..., xd, 1]
            case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
            y = y * mask
            
            
            # batch_size = x.size(0)
            if test_type == 'frames':
                pass
            elif test_type == 'accumulate': 
                if prev_case_id == -1:
                    # new case start
                    if len(preds)> 0: 
                        preds=torch.stack(preds, dim=1)   # [1, t, x1, ...,xd, v]
                        gts = torch.stack(gts, dim=1) # [1, t, x1, ...,xd, v]
                        for name in metric_names:
                            metric_fn = getattr(metrics, name)
                            cw, sw=metric_fn(preds, gts)
                            res_dict["cw_res"][name].append(cw)
                            res_dict["sw_res"][name].append(sw)
                    preds = []
                    gts = []
                else:
                    x = pred.detach().clone() # x: input tensor (The previous time step prediction) [b, x1, ..., xd, v]
            
            
            pred = model(x, case_params, mask, grid)

            if test_type == 'frames':
                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    cw, sw=metric_fn(pred, y)
                    res_dict["cw_res"][name].append(cw)
                    res_dict["sw_res"][name].append(sw)
            else: # accumulate
                preds.append(pred)
                gts.append(y) 
                
            prev_case_id = case_id

        # after the last frame
        if test_type == 'accumulate' and len(preds)>0:
            preds=torch.stack(preds, dim=1)
            gts = torch.stack(gts, dim=1) # [1, t, x1, ...,xd, v]
            for name in metric_names:
                metric_fn = getattr(metrics, name)
                cw, sw=metric_fn(preds, gts)
                res_dict["cw_res"][name].append(cw)
                res_dict["sw_res"][name].append(sw)

    # aggregate the results
    for name in metric_names:
        cw_res_list = res_dict["cw_res"][name]
        sw_res_list = res_dict["sw_res"][name]
        if name == "MaxError":
            cw_res = torch.stack(cw_res_list, dim=0)
            cw_res, _ = torch.max(cw_res, dim=0)
            sw_res = torch.stack(sw_res_list)
            sw_res = torch.max(sw_res)
        else:
            cw_res = torch.cat(cw_res_list, dim=0)
            cw_res = torch.mean(cw_res, dim=0)
            sw_res = torch.cat(sw_res_list, dim=0)
            sw_res = torch.mean(sw_res, dim=0)
        res_dict["cw_res"][name] = cw_res
        res_dict["sw_res"][name] = sw_res

    metrics.print_res(res_dict)
    metrics.write_res(res_dict, 
                      os.path.join( output_dir, args["model_name"]+"_transfering_results_"+test_type+".csv"),
                      tag =  args["flow_name"]+ "_"+args['dataset']['case_name']+"->"+target_case_name, 
                    append = True)
    return 



def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    saved_dir = os.path.join(args["saved_dir"], os.path.join(args["model_name"], args["flow_name"] + '_' + args['dataset']['case_name']))
    
    output_dir = os.path.join(args['output_dir'], 'transfering_test')


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
    
    train_data, val_data, test_data, test_ms_data = get_dataset(args)
    train_loader, val_loader, test_loader, test_ms_loader = get_dataloader(train_data, val_data, test_data, test_ms_data, args)
    channel_min, channel_max = get_min_max(train_loader)   
    args["channel_min_max"] = (channel_min, channel_max)
    if args["use_norm"]:
        print("use min_max normalization with min=", channel_min.tolist(), ", max=", channel_max.tolist())
    input, output, _, case_params, grid, _, = next(iter(val_loader))
    print("input tensor shape: ", input.shape[1:])
    print("output tensor shape: ", output.shape[1:] if val_loader.dataset.multi_step_size==1 else output.shape[2:])
    spatial_dim = grid.shape[-1]
    n_case_params = case_params.shape[-1]
    args["model"]["num_points"] = reduce(lambda x,y: x*y, grid.shape[1:-1])  # get num_points, especially of irregular geometry(point clouds)
    model = get_model(spatial_dim, n_case_params, args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args['model_name']} has {num_params} parameters")

    # loading best checkpoint
    print(f"Transfering test, loading checkpoint from {saved_path}-best.pt")
    checkpoint = torch.load(saved_path + "-best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    for target_case_name in args['dataset']['case_name_targets'].split(','):
        # get target dataset and dataloader
        target_dataset = get_test_dataset_target(args, target_case_name)
        if args["use_norm"]:
            target_dataset.apply_norm(channel_min, channel_max)
        target_loader = DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=0)
        print()
        print("transfering results for", args["flow_name"],":",  args['dataset']['case_name']+" -> "+target_case_name )
        test_loop(target_loader, model, device, output_dir, args, target_case_name, metric_names=['MSE', 'NMSE', 'MaxError'], test_type="frames")
        test_loop(target_loader, model, device, output_dir, args, target_case_name, metric_names=['MSE', 'NMSE', 'MaxError'], test_type="accumulate")
        print()
    print("Done.")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("-cs", "--case_name_source", type=str, default="", help="case_name of source domain")
    parser.add_argument("-ct", "--case_name_targets", type=str, default="", help="case_name of target domains, split by ','")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    if len(cmd_args.case_name_source) > 0:
        args['dataset']['case_name'] = cmd_args.case_name_source
    
    args['dataset']['case_name_targets'] = cmd_args.case_name_targets
    for case_name_target in args['dataset']['case_name_targets'].split(','):
        assert case_name_target != args['dataset']['case_name'], "source and target case_name should be different"

    # set use_norm
    use_norm_default=True
    if args["flow_name"] in ["TGV","Darcy"]:
        use_norm_default = False
    args["use_norm"] = args.get("use_norm", use_norm_default)

    setup_seed(args["seed"])
    print(args, flush = True)
    main(args)