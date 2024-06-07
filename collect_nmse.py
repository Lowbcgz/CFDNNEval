from utils import get_model, setup_seed
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
from dataset import *
from torch.utils.data import DataLoader
from metrics import NMSE, MSE, MaxError

default_minmax_channels = {
    "cavity": {
        # "bc": torch.tensor([[-21.820903778076172, -36.05586242675781, -291.2026672363281],
        #                     [34.55437469482422, 21.9743709564209, 871.6431274414062]]),
        # "re": torch.tensor([[-0.42775022983551025, -0.7351908683776855, -4.729204177856445],
        #                     [0.9286726117134094, 0.40977877378463745, 4.623359680175781]]),
        "ReD": torch.tensor([[-0.7218633890151978, -0.7597835659980774, -12.665838241577148],
                              [0.9996216297149658, 0.45298323035240173, 18.963367462158203]]),
        # "ReD_bc_re": torch.tensor([[-21.820903778076172, -36.05586242675781, -291.2026672363281],
        #                            [34.55437469482422, 21.9743709564209, 871.6431274414062]]),
    },
    "tube": {
        "bc": torch.tensor([[0.0, -0.23288129270076752],[1.496768832206726, 0.23283222317695618]]),
        # "geo": torch.tensor([[-0.00024760616361163557, -0.24431051313877106], [1.500606894493103, 0.24422840774059296]]),
        # "prop": torch.tensor([[0.0, -0.26001930236816406], [1.4960201978683472, 0.260026216506958]]),
        # "bc_geo": torch.tensor([[-0.00024760616361163557, -0.24431051313877106],[1.500606894493103, 0.24422840774059296]]),
        # "prop_bc": torch.tensor([[0.0, -0.26001930236816406],[1.496768832206726, 0.260026216506958]]),
        # "prop_geo": torch.tensor([[-0.00024760616361163557, -0.26001930236816406], [1.500606894493103, 0.260026216506958]]),
        # "proprop_bc_geo": torch.tensor([[-0.00024760616361163557, -0.26001930236816406],[1.500606894493103, 0.260026216506958]]),
    },
    "Darcy":{
        "darcy": torch.tensor([0.0,1.0]),
        # "PDEBench": torch.tensor([0.0,1.0])
    },
    "TGV":{
        # "single": torch.tensor([0.0,1.0]),
        "all": torch.tensor([0.0,1.0])
    },
    "NSCH":{
        "ca": torch.tensor([[-1.0031050443649292, -1.0, -0.006659443024545908],
                             [1.0169320106506348, 1.0, 0.006659443024545908]])
    },
    "cylinder":{
        "rRE": torch.tensor([[-0.873637855052948, -1.2119133472442627, -8.137107849121094],
                             [2.213331460952759, 1.185351848602295, 2.3270020484924316]])
    },
    "ircylinder":{
        "irRE": torch.tensor([[-0.9188112616539001, -1.2337069511413574, -8.151810646057129],
                              [2.2467875480651855, 1.2144097089767456, 2.6829822063446045]])
    },
    "hills":{
        "rRE": torch.tensor([ [-24.633068084716797, -29.008529663085938, -12.332571029663086, -242.58314514160156],
                             [66.42779541015625, 29.029993057250977, 25.84693717956543, 854.3807983398438]])
    },
    "irhills":{
        "irRE": torch.tensor([[-27.92310333251953, -31.912891387939453, -12.819289207458496, -313.4261779785156],
                              [66.46695709228516, 31.170812606811523, 25.986618041992188, 862.0435180664062]])
    }
}
def get_test_dataset(args):
    dataset_args = args["dataset"]
    if(args["flow_name"] == "tube"):
        test_data = TubeDataset(filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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
                                case_name=dataset_args['case_name'],
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

def plot_curves(ys, metric_name, output_dir, figname):
    fig, ax = plt.subplots()
    xs = 1+np.arange(len(ys))
    ax.plot(xs,ys)
    ax.set_ylabel(metric_name)
    ax.set_xlabel('Time step ahead')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{figname}.png")

def test_collect(test_loader, model, device, output_dir, args):
    model.eval()
    selected_case_id = 0  # set by default 
    gt_list = []
    pred_list = []

    with torch.no_grad():
        if_init = True
        #record frames
        for x, y, mask, case_params, grid, case_id in test_loader:
            case_id = case_id.item()
            if case_id != selected_case_id:
                if if_init:
                    continue
                else:
                    break
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            case_params = case_params.to(device)
            grid = grid.to(device)
            y = y * mask
            if if_init:
                if_init = False
            else:
                x = pred
            pred = model(x, case_params, mask, grid)
            pred_list.append(pred)
            gt_list.append(y)
        
        assert len(pred_list) == len(gt_list)
        total_frames = len(pred_list)
        print(f'total frames is {total_frames}')

        pred = torch.cat(pred_list, dim=0)
        gt = torch.cat(gt_list, dim=0)

        #get error (without de-normalization)
        cw, sw = MSE(pred,gt)  # (bs, c), (bs,)
        mses = torch.concat([cw, sw.unsqueeze(-1)],dim=-1)
        cw, sw = NMSE(pred,gt)
        nmses = torch.concat([cw, sw.unsqueeze(-1)],dim=-1)
        cw, sw = MaxError(pred,gt)
        max_errors = torch.concat([cw, sw.unsqueeze(-1)],dim=-1)

        print("nmses shape:", nmses.shape, " (T, num_channel+1), 1 is the global")

        if args["use_norm"]:
            (channel_min, channel_max) = args["channel_min_max"] 
            channel_min, channel_max = channel_min.to(device), channel_max.to(device)
            pred = pred*(channel_max-channel_min)+channel_min
            gt = gt*(channel_max-channel_min)+channel_min
            cw, sw = MSE(pred,gt)  # (bs, c), (bs,)
            mses_denorm = torch.concat([cw, sw.unsqueeze(-1)],dim=-1)
            cw, sw = NMSE(pred,gt)
            nmses_denorm = torch.concat([cw, sw.unsqueeze(-1)],dim=-1)
            cw, sw = MaxError(pred,gt)
            max_errors_denorm = torch.concat([cw, sw.unsqueeze(-1)],dim=-1)
            
            print("nmses_denorm shape:", nmses_denorm.shape, " (T, num_channel+1), 1 is the global")

        # saving
        np.save(os.path.join(output_dir, 'mse.npy'), mses.cpu().numpy())
        np.save(os.path.join(output_dir, 'nmse.npy'), nmses.cpu().numpy())
        np.save(os.path.join(output_dir, 'max_error.npy'), max_errors.cpu().numpy())
        if args["use_norm"]:
            np.save(os.path.join(output_dir, 'mses_denorm.npy'), mses_denorm.cpu().numpy())
            np.save(os.path.join(output_dir, 'nmses_denorm.npy'), nmses_denorm.cpu().numpy())
            np.save(os.path.join(output_dir, 'max_errors_denorm.npy'), max_errors_denorm.cpu().numpy())

        # ploting for a glance
        plot_curves(nmses[:,-1].cpu().numpy(), "NMSE", output_dir, "_".join(["nmse", args["model_name"], args["flow_name"] ,args['dataset']['case_name']]))
        if args["use_norm"]:
            plot_curves(nmses_denorm[:,-1].cpu().numpy(), "NMSE_denorm", output_dir, "_".join(["nmse-denorm", args["model_name"], args["flow_name"] ,args['dataset']['case_name']]))

                      
def main(args):
    #init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    saved_dir = os.path.join(args["saved_dir"], os.path.join(args["model_name"], args["flow_name"] + '_' + args['dataset']['case_name']))

    output_dir = os.path.join(args['output_dir'], 'collect_nmse')
    output_dir = os.path.join(output_dir, os.path.join(args["model_name"],args["flow_name"] + '_' + args['dataset']['case_name']))

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
    test_dataset = get_test_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    # set some train args
    input, output, _, case_params, grid, _, = next(iter(test_loader))
    print("input tensor shape: ", input.shape[1:])
    print("output tensor shape: ", output.shape[1:])
    spatial_dim = grid.shape[-1]
    n_case_params = case_params.shape[-1]
    args["model"]["num_points"] = reduce(lambda x,y: x*y, grid.shape[1:-1])  # get num_points, especially of irregular geometry(point clouds)

    # get min_max per channel of train-set, query from dict
    if args["use_norm"]:
        channel_min, channel_max = default_minmax_channels[args["flow_name"]][dataset_args["case_name"]]
        print("use min_max normalization with min=", channel_min.tolist(), ", max=", channel_max.tolist())
        print(channel_max, channel_min)
        args["channel_min_max"] = (channel_min, channel_max)
        test_loader.dataset.apply_norm(channel_min, channel_max)

    #model
    model = get_model(spatial_dim, n_case_params, args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args['model_name']} has {num_params} parameters")

    #load checkpoint
    checkpoint = torch.load(saved_path + "-best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    #test and plot
    print('Start_collecting...')
    test_collect(test_loader, model, device, output_dir, args)
    print(f'Saved in folder: {output_dir}')
    print("Done.", end="\n")

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

    # set use_norm
    use_norm_default=True
    if args["flow_name"] in ["TGV","Darcy"]:
        use_norm_default = False
    args["use_norm"] = args.get("use_norm", use_norm_default)
    setup_seed(args["seed"])
    print(args)
    main(args)