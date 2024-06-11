import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from model import FNO2d, FNO3d, LSM_2d, LSM_3d, AutoDeepONet, AutoDeepONet_3d, UNO2d, UNO3d, KNO2d, KNO3d, UNet2d, UNet3d, LSM_2d_ir, geoFNO2d, Oformer
from dataset import *
import os
import shutil

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
                                stable_state_diff = dataset_args['stable_state_diff'],
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
                                stable_state_diff = dataset_args['stable_state_diff'],
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
                                stable_state_diff = dataset_args['stable_state_diff'],
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
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "ircylinder":
        train_data = IRCylinderDataset(
                                filename='cylinder_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = IRCylinderDataset(
                                filename='cylinder_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
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
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = IRCylinderDataset(
                                    filename='cylinder_test.hdf5',
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
                                stable_state_diff = dataset_args['stable_state_diff'],
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
                                stable_state_diff = dataset_args['stable_state_diff'],
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
                                stable_state_diff = dataset_args['stable_state_diff'],
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
                                    stable_state_diff = dataset_args['stable_state_diff'],
                                    norm_props = dataset_args['norm_props'],
                                    multi_step_size= dataset_args['multi_step_size'],
                                    reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        else:
            test_ms_data = None
    elif args["flow_name"] == "irhills":
        train_data = IRHillsDataset(
                                filename=args['flow_name'][2:] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
        val_data = IRHillsDataset(
                                filename=args['flow_name'][2:] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                multi_step_size= dataset_args['multi_step_size'],
                                reshape_parameters=dataset_args.get('reshape_parameters', True)
                                )
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
        if dataset_args['multi_step_size'] > 1:
            test_ms_data = IRHillsDataset(
                                    filename=args['flow_name'][2:] + '_test.hdf5',
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
            elif model_name == 'OFormer':
                model = Oformer(input_ch=model_args['inputs_channel']+n_case_params,
                                output_ch=model_args['outputs_channel'],
                                n_tolx=args["model"]["num_points"],
                                multi_step_size=args["dataset"]["multi_step_size"],
                                dim=2)

        elif spatial_dim == 3:
            if model_name == "FNO":
                model = FNO3d(inputs_channel=model_args['inputs_channel'],
                              outputs_channel=model_args['outputs_channel'],
                      width = model_args['width'],
                      modes1 = model_args['modes'],
                      modes2 = model_args['modes'],
                      modes3 = model_args['modes'],
                      n_case_params = n_case_params)
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
    return model

def get_min_max(dataloader):
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

