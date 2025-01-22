import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from model import FNO2d, FNO3d, LSM_2d, LSM_3d, AutoDeepONet, AutoDeepONet_3d, UNO2d, UNO3d, KNO2d, KNO3d, UNet2d, UNet3d, LSM_2d_ir, geoFNO2d, Oformer,  NUFNO2d, NUFNO3d, NUUNet2d, NUUNet3d, FourierTransformer2DLite, My_FourierTransformer2D, My_FourierTransformer3D, Darcy_FourierTransformer2D
from dataset import *
import os
import shutil
from collections import defaultdict

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

def get_dataset(args):
    dataset_args = args["dataset"]
    if(args["flow_name"] == "tube"):
        train_data = TubeDataset(filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = TubeDataset(filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
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
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = TubeDataset(filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "cavity":
        train_data = CavityDataset(
                                filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = CavityDataset(
                                filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
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
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = CavityDataset(
                                    filename=args['flow_name'] + '_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "cylinder":
        train_data = CylinderDataset(
                                filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = CylinderDataset(
                                filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        test_data = CylinderDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = CylinderDataset(
                                    filename=args['flow_name'] + '_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "ircylinder":
        if args["model_name"] in  ["NUFNO", "NUUNet"]:
            _IRCylinderDataset = IRCylinderDataset_NUNO
        else:
            _IRCylinderDataset = IRCylinderDataset
        train_data = _IRCylinderDataset(
                                filename='cylinder_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = _IRCylinderDataset(
                                filename='cylinder_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        test_data = _IRCylinderDataset(
                                filename='cylinder_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = _IRCylinderDataset(
                                    filename='cylinder_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "NSCH":
        train_data = NSCHDataset(
                                filename='train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = NSCHDataset(
                                filename='val.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
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
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = NSCHDataset(
                                    filename='test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                    )
        else:
            test_ms_data = None
    elif args["flow_name"] == "TGV":
        train_data = TGVDataset(
                                filename='train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = TGVDataset(
                                filename='val.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
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
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = TGVDataset(
                                    filename='test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                    )
        else:
            test_ms_data = None
    elif args['flow_name'] == 'Darcy':
        filename = "2D_DarcyFlow_beta1.0_Train.hdf5" if dataset_args["case_name"] == "PDEBench" else "darcy.hdf5"
        train_data = PDEDarcyDataset(split="train",
                                    filename=filename,
                                    saved_folder= dataset_args["saved_folder"], 
                                    reduced_batch=dataset_args["reduced_batch"],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True))
        val_data = PDEDarcyDataset(split="val", 
                                    filename=filename,
                                    saved_folder= dataset_args["saved_folder"], 
                                    reduced_batch=dataset_args["reduced_batch"],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True))
        test_data = PDEDarcyDataset(split="test", 
                                    filename=filename,
                                    saved_folder= dataset_args["saved_folder"], 
                                    reduced_batch=dataset_args["reduced_batch"],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True))
        test_ms_data = None
    elif args["flow_name"] == "hills":
        train_data = HillsDataset(
                                filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = HillsDataset(
                                filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        test_data = HillsDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = HillsDataset(
                                    filename=args['flow_name'] + '_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "irhills":
        if args["model_name"] in ["NUFNO", "NUUNet"]:
            _IRHillsDataset = IRHillsDataset_NUNO
        else:
            _IRHillsDataset = IRHillsDataset
        train_data = _IRHillsDataset(
                                filename=args['flow_name'][2:] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = _IRHillsDataset(
                                filename=args['flow_name'][2:] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        test_data = _IRHillsDataset(
                                filename=args['flow_name'][2:] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                norm_props = dataset_args['norm_props'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = _IRHillsDataset(
                                    filename=args['flow_name'][2:] + '_test.hdf5',
                                    saved_folder=dataset_args['saved_folder'],
                                    case_name=dataset_args['case_name'],
                                    reduced_resolution=dataset_args["reduced_resolution"],
                                    reduced_batch=dataset_args["reduced_batch"],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    else:
        raise ValueError("Invalid flow name.")
    print("#Train data: ", len(train_data))
    print("#Validation data: ", len(val_data))
    print("#Test data: ", len(test_data))
    return train_data, val_data, test_data, test_ms_data

def get_dataloader(train_data, val_data, test_data, test_ms_data, args):
    dataloader_args = args["dataloader"]
    if dataloader_args['num_workers'] > 0:
        train_loader = DataLoader(train_data, shuffle=True, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), 
                                batch_size=dataloader_args['train_batch_size'], 
                                num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
        val_loader = DataLoader(val_data, shuffle=False, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), 
                                batch_size=dataloader_args['val_batch_size'],
                                num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
    else:
        train_loader = DataLoader(train_data, shuffle=True,
                                batch_size=dataloader_args['train_batch_size'],
                                num_workers= 0, pin_memory=dataloader_args['pin_memory'])
        val_loader = DataLoader(val_data, shuffle=False,
                                batch_size=dataloader_args['val_batch_size'],
                                num_workers= 0, pin_memory=dataloader_args['pin_memory'])
    
    test_loader = DataLoader(test_data, shuffle=False, drop_last=True,
                            batch_size=dataloader_args['test_batch_size'],
                            num_workers= 0, pin_memory=dataloader_args['pin_memory'])
    if test_ms_data is not None:
        test_ms_loader = DataLoader(test_ms_data, shuffle=False, drop_last=True,
                                batch_size=dataloader_args['test_batch_size'],
                                num_workers=0, pin_memory=dataloader_args['pin_memory'])
    else:
        test_ms_loader = None
    
    return train_loader, val_loader, test_loader, test_ms_loader

def get_model(spatial_dim, n_case_params, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_name = args["model_name"]
    model_args = args["model"]
    if args['flow_name'] in ["Darcy"]:   # time irrelevant
        if spatial_dim == 2:
            if model_name == "FNO": 
                model = FNO2d(inputs_channel=model_args['inputs_channel'],
                                outputs_channel=model_args['outputs_channel'],
                        width = model_args['width'],
                        modes1 = model_args['modes'],
                        modes2 = model_args['modes'],
                        n_case_params = n_case_params)
            elif model_name == "LSM":
                model = LSM_2d(inputs_channel=model_args['inputs_channel'],
                        outputs_channel=model_args['outputs_channel'],
                        d_model = model_args['width'],
                        num_token=model_args['num_token'], 
                        num_basis=model_args['num_basis'], 
                        patch_size=model_args['patch_size'],
                        padding=model_args['padding'],
                        n_case_params = n_case_params)
            elif model_name == "AutoDeepONet":
                model = AutoDeepONet(
                        branch_dim= model_args["h"]*model_args["w"]*model_args['inputs_channel']+n_case_params,
                        trunk_dim =2, # (x,y)
                        out_channel=model_args['outputs_channel'],
                        width=model_args["width"],
                        trunk_depth=model_args["trunk_depth"],
                        branch_depth=model_args["branch_depth"],
                        act_name=model_args["act_fn"],
                        autoregressive_mode=False
                        )
            elif model_name == 'UNO':
                model = UNO2d(in_channels=model_args["in_channels"],
                          out_channels = model_args["out_channels"],
                          width=model_args["width"],
                          n_case_params = n_case_params)
            elif model_name == 'UNet':
                model = UNet2d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params)
            elif model_name == 'OFormer':
                model = Oformer(input_ch=model_args['inputs_channel']+n_case_params,
                                output_ch=model_args['outputs_channel'],
                                n_tolx=args["model"]["num_points"],
                                multi_step_size=args["dataset"]["multi_step_size"],
                                dim=2)
            elif model_name == 'GFormer':
                print('model:load GFormer------------------')
                subsample_nodes=3
                subsample_attn=6
                if args['dataset']['case_name'] == 'PDEBench':
                    n_grid = 128
                    n_x, n_y = 128, 128
                else:
                    n_grid = 128
                    n_x, n_y = 128, 128
                fine_grid = (n_grid-1) * subsample_nodes + 1  # 逆过来求一下原始网格数(原文这里为421的原始网格)
                n_grid_c = int(((fine_grid - 1)/subsample_attn) + 1)

                args['n_x'] = n_x
                args['n_y'] = n_y
                model_config = args['model']
                print('n_grid, n_grid_c', n_grid, n_grid_c)
                # 计算升降比例
                n_f, n_c = n_grid, n_grid_c
                factor = np.sqrt(n_c/n_f)
                factor = np.round(factor, 4)
                last_digit = float(str(factor)[-1])
                factor = np.round(factor, 3)
                if last_digit < 5:
                    factor += 5e-3
                factor = int(factor/5e-3 + 5e-1 ) * 5e-3
                down_factor = (factor, factor)
                n_m = round(n_f*factor)-1
                up_size = ((n_m, n_m), (n_f, n_f))
                downsample, upsample =  down_factor, up_size
                # 使用原始网格计算模型中的降采样、升采样大小
                # downsample, upsample = get_scaler_sizes(n_grid, n_grid_c)
                model_config = args['model']
                model_config['downscaler_size'] = downsample
                print('downsample:', downsample)
                model_config['upscaler_size'] = upsample
                model_config['attn_norm'] = not model_config['attn_norm']  # True
                model_config['node_feats'] = int(model_config['node_feats'])
                if model_config['attention_type'] == 'fourier' or n_grid < 211:
                    model_config['norm_eps'] = 1e-7
                elif model_config['attention_type'] == 'galerkin' and n_grid >= 211:
                    model_config['norm_eps'] = 1e-5

                model = Darcy_FourierTransformer2D(**model_config)
   
        else:
            #TODO
            pass
    else:
        if spatial_dim == 1:
            pass
        elif spatial_dim == 2:
            if model_name == "FNO":
                model = FNO2d(inputs_channel=model_args['inputs_channel'],
                              outputs_channel=model_args['outputs_channel'],
                      width = model_args['width'],
                      modes1 = model_args['modes'],
                      modes2 = model_args['modes'],
                      n_case_params = n_case_params)
            elif model_name == "NUFNO":
                model = NUFNO2d(inputs_channel=model_args['inputs_channel'],
                              outputs_channel=model_args['outputs_channel'],
                      width = model_args['width'],
                      modes1 = model_args['modes'],
                      modes2 = model_args['modes'],
                      n_case_params = n_case_params,
                      n_subdomains = model_args['n_subdomains'])
            elif model_name == "geoFNO":
                model = geoFNO2d(inputs_channel=model_args['inputs_channel'],
                                outputs_channel=model_args['outputs_channel'],
                        width = model_args['width'],
                        modes1 = model_args['modes'],
                        modes2 = model_args['modes'],
                        n_case_params = n_case_params)
            elif model_name == "LSM":
                if model_args["irregular_geo"]:
                    model = LSM_2d_ir(
                            inputs_channel=model_args['inputs_channel'],
                            outputs_channel=model_args['outputs_channel'],
                            d_model = model_args['width'],
                            num_token=model_args['num_token'],
                            num_basis=model_args['num_basis'],
                            patch_size=model_args['patch_size'],
                            use_iphi = model_args['use_iphi'],
                            n_case_params = n_case_params
                    )
                else: 
                    model = LSM_2d(inputs_channel=model_args['inputs_channel'],
                            outputs_channel=model_args['outputs_channel'],
                        d_model = model_args['width'],
                        num_token=model_args['num_token'], 
                        num_basis=model_args['num_basis'], 
                        patch_size=model_args['patch_size'],
                        padding=model_args['padding'],
                        n_case_params = n_case_params)
            elif model_name == "AutoDeepONet":
                if model_args["irregular_geo"]:
                    model = AutoDeepONet(
                            branch_dim= model_args["num_points"]*model_args['inputs_channel']+n_case_params,
                            trunk_dim =2, # (x,y)
                            out_channel=model_args['outputs_channel'],
                            width=model_args["width"],
                            trunk_depth=model_args["trunk_depth"],
                            branch_depth=model_args["branch_depth"],
                            act_name=model_args["act_fn"],
                            irregular_geometry=True
                            )
                else:
                    model = AutoDeepONet(
                            branch_dim= model_args["h"]*model_args["w"]*model_args['inputs_channel']+n_case_params,
                            trunk_dim =2, # (x,y)
                            out_channel=model_args['outputs_channel'],
                            width=model_args["width"],
                            trunk_depth=model_args["trunk_depth"],
                            branch_depth=model_args["branch_depth"],
                            act_name=model_args["act_fn"],
                            )
            elif model_name == 'UNO':
                model = UNO2d(in_channels=model_args["in_channels"],
                          out_channels = model_args["out_channels"],
                          width=model_args["width"],
                          n_case_params = n_case_params)
            elif model_name == 'KNO':
                model_args_copy = model_args.copy()
                model_args_copy.pop('num_points')
                model = KNO2d(n_case_params=n_case_params, **model_args_copy)
            elif model_name == 'UNet':
                model = UNet2d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params)
            elif model_name == 'NUUNet':
                model = NUUNet2d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params,
                        n_subdomains = model_args['n_subdomains'])
            elif model_name == 'OFormer':
                model = Oformer(input_ch=model_args['inputs_channel']+n_case_params,
                                output_ch=model_args['outputs_channel'],
                                n_tolx=args["model"]["num_points"],
                                multi_step_size=args["dataset"]["multi_step_size"],
                                dim=2)
            elif model_name == 'GFormer':
                model_config = args["model"]
                # model 1：lite
                if model_config['model_type']=='lite':
                    config = defaultdict(lambda: None,
                                #  node_feats=10+2,
                                    node_feats=int(model_config['node_feats']),  # 重新设置12+2
                                    pos_dim=2,
                                    out_dim = model_config['out_dim'],
                                    n_targets=1,
                                    n_hidden=128,  # attention's d_model
                                    num_feat_layers=0,
                                    num_encoder_layers=6,
                                    n_head=4,
                                    dim_feedforward=256,
                                    attention_type='galerkin',
                                    feat_extract_type=None,
                                    xavier_init=0.01,
                                    diagonal_weight=0.01,
                                    layer_norm=True,
                                    attn_norm=False,
                                    return_attn_weight=False,
                                    return_latent=False,
                                    decoder_type='ifft',
                                    freq_dim=20,  # hidden dim in the frequency domain
                                    num_regressor_layers=2,  # number of spectral layers
                                    fourier_modes=12,  # number of Fourier modes
                                    spacial_dim=2,
                                    spacial_fc=False,
                                    dropout=0.0,
                                    encoder_dropout=0.0,
                                    decoder_dropout=0.0,
                                    ffn_dropout=0.05,
                                    debug=False,
                                    )
                    model = FourierTransformer2DLite(**config)
                # model 2：normal
                else:
                    # add original setup
                    subsample_nodes=3,
                    subsample_attn=6,
                    model_config['node_feats'] = int(model_config['node_feats'])
                    model_config['norm_eps'] = 1e-5
                    model = My_FourierTransformer2D(**model_config)

        elif spatial_dim == 3:
            if model_name == "FNO":
                model = FNO3d(inputs_channel=model_args['inputs_channel'],
                              outputs_channel=model_args['outputs_channel'],
                      width = model_args['width'],
                      modes1 = model_args['modes'],
                      modes2 = model_args['modes'],
                      modes3 = model_args['modes'],
                      n_case_params = n_case_params)
            elif model_name == "NUFNO":
                model = NUFNO3d(inputs_channel=model_args['inputs_channel'],
                              outputs_channel=model_args['outputs_channel'],
                      width = model_args['width'],
                      modes1 = model_args['modes'],
                      modes2 = model_args['modes'],
                      modes3 = model_args['modes'],
                      n_case_params = n_case_params,
                      n_subdomains = model_args['n_subdomains'])
            elif model_name == "LSM":
                if model_args["irregular_geo"]:
                    raise NotImplementedError("LSM for 3D is not implemented for irregular geometry yet.")
                else: 
                    model = LSM_3d(inputs_channel=model_args['inputs_channel'],
                            outputs_channel=model_args['outputs_channel'],
                        d_model = model_args['width'],
                        num_token=model_args['num_token'], 
                        num_basis=model_args['num_basis'], 
                        patch_size=model_args['patch_size'],
                        padding=model_args['padding'],
                        n_case_params = n_case_params)
            elif model_name ==model_name == "AutoDeepONet":
                if model_args["irregular_geo"]:
                    model = AutoDeepONet_3d(
                            branch_dim= model_args["num_points"]*model_args['inputs_channel']+n_case_params,
                            trunk_dim =3, # (x,y,z)
                            out_channel=model_args['outputs_channel'],
                            width=model_args["width"],
                            trunk_depth=model_args["trunk_depth"],
                            branch_depth=model_args["branch_depth"],
                            act_name=model_args["act_fn"],
                            irregular_geometry=True
                            )
                else:
                    model = AutoDeepONet_3d(
                            branch_dim= model_args["h"]*model_args["w"]*model_args["d"]*model_args['inputs_channel']+n_case_params,
                            trunk_dim =3, # (x,y,z)
                            out_channel=model_args['outputs_channel'],
                            width=model_args["width"],
                            trunk_depth=model_args["trunk_depth"],
                            branch_depth=model_args["branch_depth"],
                            act_name=model_args["act_fn"],
                            )
            elif model_name == 'UNO':
                model = UNO3d(in_channels=model_args["in_channels"],
                          out_channels = model_args["out_channels"],
                          width=model_args["width"],
                          n_case_params = n_case_params)
            elif model_name == 'KNO':
                model_args_copy = model_args.copy()
                model_args_copy.pop('num_points')
                model = KNO3d(n_case_params=n_case_params, **model_args_copy)
            elif model_name == 'UNet':
                model = UNet3d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params)
            elif model_name == 'NUUNet':
                model = NUUNet3d(in_channels=model_args['in_channels'],
                        out_channels=model_args['out_channels'],
                        init_features=model_args['init_features'],
                        n_case_params = n_case_params,
                        n_subdomains = model_args['n_subdomains'])
            elif model_name == 'GFormer':
                model_config = args["model"]
                # 3D for hills
                model_config['attn_norm'] = not model_config['attn_norm']
                model_config['node_feats'] = int(model_config['node_feats'])
                model_config['norm_eps'] = 1e-5

                model = My_FourierTransformer3D(**model_config)
            elif model_name == 'OFormer':
                model = Oformer(input_ch=model_args['inputs_channel']+n_case_params,
                                output_ch=model_args['outputs_channel'],
                                n_tolx=args["model"]["num_points"],
                                multi_step_size=args["dataset"]["multi_step_size"],
                                dim=3)

    return model

default_minmax_channels = {
    "cavity": {
        "bc": torch.tensor([[-21.820903778076172, -36.05586242675781, -291.2026672363281],
                            [34.55437469482422, 21.9743709564209, 871.6431274414062]]),
        "re": torch.tensor([[-0.42775022983551025, -0.7351908683776855, -4.729204177856445],
                            [0.9286726117134094, 0.40977877378463745, 4.623359680175781]]),
        "ReD": torch.tensor([[-0.7218633890151978, -0.7597835659980774, -12.665838241577148],
                              [0.9996216297149658, 0.45298323035240173, 18.963367462158203]]),
        "ReD_bc_re": torch.tensor([[-21.820903778076172, -36.05586242675781, -291.2026672363281],
                                   [34.55437469482422, 21.9743709564209, 871.6431274414062]]),
    },
    "tube": {
        "bc": torch.tensor([[0.0, -0.23288129270076752],[1.496768832206726, 0.23283222317695618]]),
        "geo": torch.tensor([[-0.00024760616361163557, -0.24431051313877106], [1.500606894493103, 0.24422840774059296]]),
        "prop": torch.tensor([[0.0, -0.26001930236816406], [1.4960201978683472, 0.260026216506958]]),
        "bc_geo": torch.tensor([[-0.00024760616361163557, -0.24431051313877106],[1.500606894493103, 0.24422840774059296]]),
        "prop_bc": torch.tensor([[0.0, -0.26001930236816406],[1.496768832206726, 0.260026216506958]]),
        "prop_geo": torch.tensor([[-0.00024760616361163557, -0.26001930236816406], [1.500606894493103, 0.260026216506958]]),
        "prop_bc_geo": torch.tensor([[-0.00024760616361163557, -0.26001930236816406],[1.500606894493103, 0.260026216506958]]),
    },
    "NSCH":{
        "ca": torch.tensor([[-1.0013279914855957, -1.0, -0.01557231042534113],
                             [1.0209170579910278, 1.0, 0.01557231042534113]]),
        "phi": torch.tensor([[-1.1152399778366089, -1.0675870180130005, -0.07914472371339798],
                             [1.0359419584274292, 1.0676339864730835, 0.07638972252607346]]),
        "eps": torch.tensor([[-1.0, -1.0, -0.029226046055555344],
                             [1.0413249731063843, 1.0, 0.029226046055555344]]),
        "mob": torch.tensor([[-1.0272589921951294, -1.0059770345687866, -0.10580600053071976],
                             [1.043949007987976, 1.0059770345687866, 0.10580600053071976]]),
        "re": torch.tensor([[-1.0012580156326294, -1.0, -0.05708836019039154],
                            [1.0202419757843018, 1.0, 0.05708836019039154]]),
        "ibc": torch.tensor([[-1.083219051361084, -9.982743263244629, -0.06129448860883713],
                             [1.119168996810913, 9.982743263244629, 0.06129448860883713]]),
    },
    "cylinder":{
        "rBC": torch.tensor([[-41.48142623901367, -64.7027359008789, -5329.173828125],
                             [110.315185546875, 63.08414077758789, 2628.212646484375]]),
        "rRE": torch.tensor([[-0.873637855052948, -1.2119133472442627, -8.137107849121094],
                             [2.213331460952759, 1.185351848602295, 2.3270020484924316]])
    },
    "ircylinder":{
        "irBC": torch.tensor([[-42.211341857910156, -65.98857879638672, -5953.85888671875],
                             [113.3804702758789, 64.10173797607422, 2933.802734375]]),
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

def get_min_max(dataloader, args):
    flow_name = args["flow_name"]
    case_name = args["dataset"]["case_name"]
    if flow_name in default_minmax_channels.keys() and case_name in default_minmax_channels[flow_name].keys():
        # get from the cache
        channel_min, channel_max = default_minmax_channels[flow_name][case_name]
    else:
        # generate online
        for i, batch in enumerate(dataloader):
            x = batch[0] # inputs [bs, h, w, c] or [bs, nx, c]
            c = x.shape[-1]
            if i == 0:  # initialize
                channel_min, _ = x.view(-1, c).min(dim=0)
                channel_max, _ = x.view(-1, c).max(dim=0)
            else:
                batch_max_value, _ = x.view(-1,c).max(dim=0)
                batch_min_value, _ = x.view(-1,c).min(dim=0)
                channel_min = torch.minimum(channel_min, batch_min_value)
                channel_max = torch.maximum(channel_max, batch_max_value)
    return channel_min, channel_max


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = None):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """
    # save checkpoint
    torch.save(state, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def ensure_dir(dir_name: str):
    """Creates folder if not exists.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

