import numpy as np
import torch
import random

from dataset import TubeDataset, NSCHDataset, PDEDarcyDataset, CavityDataset, TGVDataset, IRCylinderDataset, IRHillsDataset, HillsDataset, CylinderDataset
from model import MPNN2D, MPNN3D, GNOT, MPNNIrregular

def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn


def get_dataset(args):
    dataset_args = args["dataset"].copy()
    if args["flow_name"] == "tube":
        train_dataset = TubeDataset(filename="tube_train.hdf5", **dataset_args)
        val_dataset = TubeDataset(filename="tube_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = TubeDataset(filename="tube_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = TubeDataset(filename="tube_test.hdf5", **dataset_args)
    elif args["flow_name"] == "NSCH":
        train_dataset = NSCHDataset(filename="train.hdf5", **dataset_args)
        val_dataset = NSCHDataset(filename="val.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = NSCHDataset(filename="test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = NSCHDataset(filename="test.hdf5", **dataset_args)
    elif args["flow_name"] == "Darcy":
        if dataset_args["case_name"] == "PDEBench":
            dataset_args.pop("case_name")
            train_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="train", **dataset_args)
            val_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="val", **dataset_args)
            test_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="test", **dataset_args)
        elif dataset_args["case_name"] == "darcy":
            dataset_args.pop("case_name")
            train_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="train", **dataset_args)
            val_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="val", **dataset_args)
            test_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="test", **dataset_args)
        else:
            raise NotImplementedError
    elif args["flow_name"] == "cavity":
        train_dataset = CavityDataset(filename="cavity_train.hdf5", **dataset_args)
        val_dataset = CavityDataset(filename="cavity_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = CavityDataset(filename="cavity_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = CavityDataset(filename="cavity_test.hdf5", **dataset_args)
    elif args["flow_name"] == "TGV":
        train_dataset = TGVDataset(filename="train.hdf5", **dataset_args)
        val_dataset = TGVDataset(filename="val.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = TGVDataset(filename="test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = TGVDataset(filename="cavity_test.hdf5", **dataset_args)
    elif args["flow_name"] == "cylinder":
        train_dataset = IRCylinderDataset(filename="cylinder_train.hdf5", **dataset_args)
        val_dataset = IRCylinderDataset(filename="cylinder_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = IRCylinderDataset(filename="cylinder_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = IRCylinderDataset(filename="cylinder_test.hdf5", **dataset_args)
    elif args["flow_name"] == "regular_cylinder":
        train_dataset = CylinderDataset(filename="cylinder_train.hdf5", **dataset_args)
        val_dataset = CylinderDataset(filename="cylinder_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = CylinderDataset(filename="cylinder_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = CylinderDataset(filename="cylinder_test.hdf5", **dataset_args)
    elif args["flow_name"] == "hills":
        train_dataset = IRHillsDataset(filename="hills_train.hdf5", **dataset_args)
        val_dataset = IRHillsDataset(filename="hills_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = IRHillsDataset(filename="hills_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = IRHillsDataset(filename="hills_test.hdf5", **dataset_args)
    elif args["flow_name"] == "regular_hills":
        train_dataset = HillsDataset(filename="hills_train.hdf5", **dataset_args)
        val_dataset = HillsDataset(filename="hills_dev.hdf5", **dataset_args)
        if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
            dataset_args.pop("multi_step_size")
            test_dataset = HillsDataset(filename="hills_test.hdf5", multi_step_size=1, **dataset_args)
        else:
            test_dataset = HillsDataset(filename="hills_test.hdf5", **dataset_args)
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset


def get_model(args):
    model_args = args["model"].copy()
    if args["model_name"] == "mpnn":
        spatial_dim = model_args.pop("spatial_dim")
        if spatial_dim == 2:
            model = MPNN2D(**model_args)
        elif spatial_dim == 3:
            model = MPNN3D(**model_args)
        else:
            raise NotImplementedError
    elif args["model_name"] == "mpnn_irregular":
        model = MPNNIrregular(**model_args)
    elif args["model_name"] == "gnot":
        model = GNOT(**model_args)
    else:
        raise NotImplementedError
    return model


def get_model_name(args):
    suffix = (f"_lr{args['optimizer']['lr']}" + 
              f"_bs{args['dataloader']['batch_size']}" + 
              f"_wd{args['optimizer']['weight_decay']}" +
              f"_ep{args['epochs']}")
    if args["model_name"] == "mpnn" or args["model_name"] == "mpnn_irregular":
        model_name = (f"{args['model_name']}" +
                      f"_k{args['model']['neighbors']}" +
                      f"_layer{args['model']['hidden_layers']}" +
                      f"_dim{args['model']['hidden_features']}" +
                      f"_v{args['model']['var_id']}")
        return model_name + suffix
    elif args["model_name"] == "gnot":
        model_name = (f"{args['model_name']}" +
                      f"_layer{args['model']['n_layers']}" +
                      f"_dim{args['model']['n_hidden']}" +
                      f"_head{args['model']['n_head']}")
        return model_name + suffix
    else:
        raise NotImplementedError
    

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


def get_test_dataset(args):
    # get dataset arguments
    dataset_args = args["dataset"].copy()
    if "multi_step_size" in dataset_args and dataset_args["multi_step_size"] > 1:
        dataset_args["multi_step_size"] = 1
        
    # create dataset
    if args["flow_name"] == "tube":
        test_dataset = TubeDataset(filename="tube_test.hdf5", **dataset_args)
    elif args["flow_name"] == "NSCH":
        test_dataset = NSCHDataset(filename="test.hdf5", **dataset_args)
    elif args["flow_name"] == "Darcy":
        if dataset_args["case_name"] == "PDEBench":
            dataset_args.pop("case_name")
            test_dataset = PDEDarcyDataset(filename="2D_DarcyFlow_beta1.0_Train.hdf5", split="test", **dataset_args)
        elif dataset_args["case_name"] == "darcy":
            dataset_args.pop("case_name")
            test_dataset = PDEDarcyDataset(filename="darcy.hdf5", split="test", **dataset_args)
        else:
            raise NotImplementedError
    elif args["flow_name"] == "cavity":
        test_dataset = CavityDataset(filename="cavity_test.hdf5", **dataset_args)
    elif args["flow_name"] == "TGV":
        test_dataset = TGVDataset(filename="test.hdf5", **dataset_args)
    elif args["flow_name"] == "regular_cylinder":
        test_dataset = CylinderDataset(filename="cylinder_test.hdf5", **dataset_args)
    elif args["flow_name"] == "cylinder":
        test_dataset = IRCylinderDataset(filename="cylinder_test.hdf5", **dataset_args)
    elif args["flow_name"] == "regular_hills":
        test_dataset = HillsDataset(filename="hills_test.hdf5", **dataset_args)
    elif args["flow_name"] == "hills":
        test_dataset = IRHillsDataset(filename="hills_test.hdf5", **dataset_args)
    else:
        raise NotImplementedError
    return test_dataset