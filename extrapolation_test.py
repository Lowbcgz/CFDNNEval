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

data_root = "/data1/FluidData/extrapolation/"

def errors_and_draw(preds, gts , fig_dir, error_dir, args):
    # pred = pred * mask
    # pred = pred * (min_max[1] - min_max[0]) + min_max[0]

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    assert len(preds) == len(gts)
    #plot only for regular grid
    if args["dataset"]["case_name"] == "rRE":  
        cnt = 0
        time_list = ['0', '0.25T', '0.5T', '0.75T', 'T']
        for i in range(0, len(preds), len(preds) // 4):
            pred = preds[i]
            gt = gts[i]
            time = time_list[cnt]
            vel_label = gt[..., 0:2].norm(dim=-1)
            vel_pred = pred[..., 0:2].norm(dim=-1)
            plot_predictions(label = vel_label, pred = vel_pred, out_dir=Path(fig_dir), message=f'vel_at_time_' + time)
            cnt += 1
        
        if cnt != 5:
            pred = preds[-1]
            gt = gts[-1]
            time = time_list[-1]
            vel_label = gt[..., 0:2].norm(dim=-1)
            vel_pred = pred[..., 0:2].norm(dim=-1)
            plot_predictions(label = vel_label, pred = vel_pred, out_dir=Path(fig_dir), message=f'vel_at_time_' + time)
    
    #get error
    mse = []
    nmse = []
    max_error = []

    for i in range(len(preds)):
        pred = preds[i]
        gt = gts[i]

        nc = pred.shape[-1]
        pred = pred.reshape(-1, nc).cpu().detach().numpy()
        gt = gt.reshape(-1, nc).cpu().detach().numpy()

        error = pred - gt
        mse.append(np.mean(error ** 2, axis=0))
        nmse.append(np.mean(error ** 2, axis=0) / np.mean(gt ** 2, axis = 0))
        max_error.append(np.max(np.abs(error), axis=0))

    np.save(os.path.join(error_dir, 'mse.npy'), np.array(mse))
    np.save(os.path.join(error_dir, 'nmse.npy'), np.array(nmse))
    np.save(os.path.join(error_dir, 'max_error.npy'), np.array(max_error))

def test_loop(test_loader, model, device, fig_dir, error_dir, args, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], test_type = 'frames'):
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
    
    ckpt_dir = "./test/" + test_type + '/' + args["flow_name"] + '_' + args['dataset']['case_name']
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

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
                        
                        # TODO : call drawing here
                        fig_dir_sub = Path(fig_dir) / f"case{case_count-1}"
                        error_dir_sub = Path(error_dir) / f"case{case_count-1}"
                        errors_and_draw(preds.squeeze(0), gts.squeeze(0), fig_dir_sub, error_dir_sub, args)
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
            
            # TODO : call drawing here
            fig_dir_sub = Path(fig_dir) / f"case{case_count}"
            error_dir_sub = Path(error_dir) / f"case{case_count}"
            errors_and_draw(preds.squeeze(0), gts.squeeze(0), fig_dir_sub, error_dir_sub, args)

    if test_type == "accumulate": # print results of each case
        for i in range(case_count):
            sub_res = {"cw_res":{},  # channel-wise
                    "sw_res":{},}  # sample-wise
            for name in metric_names:
                sub_res["cw_res"][name] = res_dict["cw_res"][name][i].squeeze(0)
                sub_res["sw_res"][name] = res_dict["sw_res"][name][i].squeeze(0)
            print(f"For case {i} result: ")
            metrics.print_res(sub_res)
            print("-"*30)
            metrics.write_res(sub_res, 
                        os.path.join( args["output_dir"],"extp_test_"+args["model_name"]+args["flow_name"] + '_results.csv'),
                        tag =  test_type + "_"+ args['dataset']['case_name']+f"_case{1+i}", 
                        append = True)

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

    print("aggregated results")
    metrics.print_res(res_dict)
    metrics.write_res(res_dict, 
                      os.path.join( args["output_dir"],"extp_test_"+args["model_name"]+args["flow_name"] + '_results.csv'),
                      tag =  test_type + "_"+ args['dataset']['case_name']+"_aggreate", 
                    append = True)
    return 



def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    saved_dir = os.path.join(args["saved_dir"], os.path.join(args["model_name"], args["flow_name"] + '_' + args['dataset']['case_name']))
    
    fig_dir = os.path.join(args['output_dir'], 'extp_fig')
    fig_dir = os.path.join(fig_dir, os.path.join(args["model_name"],args["flow_name"] + '_' + args['dataset']['case_name']))

    error_dir = os.path.join(args['output_dir'], 'extp_error')
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
    print(f"Extrapolation test, loading checkpoint from {saved_path}-best.pt")
    checkpoint = torch.load(saved_path + "-best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # get extrapolation dataset
    if args["flow_name"] == "cylinder":
        from utils import CylinderDataset
        extp_dataset = CylinderDataset(
                                filename= 'cylinder_extarpolation.hdf5',
                                saved_folder=data_root,
                                case_name=dataset_args['case_name'], # rRE
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= 1,
                                reshape_parameters=dataset_args.get('reshape_parameters', True),
                                filter_outlier=False
                                )
        extp_dataset.apply_norm(channel_min, channel_max)
    elif args["flow_name"] == "ircylinder":
        from utils import IRCylinderDataset
        extp_dataset = IRCylinderDataset(
                                filename= 'cylinder_extarpolation.hdf5',
                                saved_folder=data_root,
                                case_name=dataset_args['case_name'], # irRE
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= 1,
                                reshape_parameters=dataset_args.get('reshape_parameters', True),
                                filter_outlier=False
                                )
        extp_dataset.apply_norm(channel_min, channel_max)
    
    extp_loader = DataLoader(extp_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loop(extp_loader, model, device, fig_dir, error_dir, args, metric_names=['MSE', 'NMSE', 'MaxError'], test_type="frames")
    test_loop(extp_loader, model, device, fig_dir, error_dir, args, metric_names=['MSE', 'NMSE', 'MaxError'], test_type="accumulate")
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("-c", "--case_name", type=str, default="", help="For the case, if no value is entered, the yaml file shall prevail")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    if len(cmd_args.case_name) > 0:
        args['dataset']['case_name'] = cmd_args.case_name

    dataset_args= args["dataset"]
    assert args["flow_name"] in ["cylinder", "ircylinder"]
    assert "RE" in dataset_args['case_name']
    args["use_norm"] = args.get("use_norm", True)

    setup_seed(args["seed"])
    print(args, flush = True)
    main(args)