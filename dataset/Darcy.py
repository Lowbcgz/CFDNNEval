import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class DarcyDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'bc',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 num_samples_max = -1,
                 multi_step_size = 1,
                 ):
        
        
        '''
        Args:
            filename(str): the file name of dataset, such as "tube_dev.hdf5"
            saved_folder(str) : The path to the folder where the dataset is stored , "/home/data2/cfdbench_hdf5/tube/"
            case_name(str): Decide what type of dataset to use, for instance, "bc", "geo", "bc_geo" ...
            reduced_resolution(int): reduced spatial resolution, default:1
            reduced_batch(int): reduced batch, default:1 
            multi_step_size(int): default: 1
        Returns:
            Note that network input does not contain case parameters in Darcy flow.
            input, label, mask, case_params(empty), self.grid, case_id
        shape:
            (x, y, c), (x, y, c), (x, y, 1), (0), (x, y, 2), (1)
        '''

        self.multi_step_size = multi_step_size

        root_path = os.path.join(saved_folder, filename)
        if_init = True
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            for name in f.keys():
                if name in case_name:
                    data_group = f[name]
                    if if_init:
                        self.inputs = np.array(data_group['input'], dtype=np.float32)[::reduced_batch, ::reduced_resolution, ::reduced_resolution]
                        self.labels = np.array(data_group['label'], dtype=np.float32)[::reduced_batch, ::reduced_resolution, ::reduced_resolution]
                        self.grids = np.array(data_group['grid'], dtype=np.float32)[::reduced_batch, ::reduced_resolution, ::reduced_resolution]
                        if_init = False
                    else:
                        self.inputs = np.concatenate((self.inputs, np.array(data_group['input'], dtype=np.float32)[::reduced_batch, ::reduced_resolution, ::reduced_resolution]))
                        self.labels = np.concatenate((self.labels, np.array(data_group['label'], dtype=np.float32)[::reduced_batch, ::reduced_resolution, ::reduced_resolution]))
                        self.grids = np.concatenate((self.grids, np.array(data_group['grid'], dtype=np.float32)[::reduced_batch, ::reduced_resolution, ::reduced_resolution]))

        self.inputs = torch.tensor(self.inputs).float()
        self.labels = torch.tensor(self.labels).float()
        self.grids = torch.tensor(self.grids).float()  

        _, x, y, _ = self.grids.shape
        b = self.inputs.shape[0]
        
        self.masks = torch.ones((b, x, y, 1)).float()
        self.case_ids = np.linspace(0, self.grids.shape[0] - 1, self.grids.shape[0], endpoint=True, dtype=np.int32)
        
        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.inputs.shape[0])
        else:
            num_samples_max = self.inputs.shape[0]
        
        self.inputs = self.inputs[:num_samples_max, ...]
        self.labels = self.labels[:num_samples_max, ...]
        self.case_ids = self.case_ids[:num_samples_max, ...]
        self.masks = self.masks[:num_samples_max, ...]
        self.grids = self.grids[:num_samples_max, ...]


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]  # (x, y, 2)
        label = self.labels[idx]  # (x, y, 2)
        mask = self.masks[idx] # (x, y, 1)
        case_id = self.case_ids[idx]
        grid = self.grids[idx]
        return inputs, label, mask, torch.empty((0)), grid, case_id



class PDEDarcyDataset(Dataset):

    beta = 1. # fix

    def __init__(self, 
                 filename="2D_DarcyFlow_beta1.0_Train.hdf5",
                 saved_folder="/data1/FluidData/darcy",
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 split="train"
                 ):
        """Dataset for Darcy flow in PDEBench.
        Args:
            filename (str): The file name of dataset, default: "2D_DarcyFlow_beta1.0_Train.hdf5".
            saved_folder (str) : The path to the folder where the dataset is stored such as "/data1/FluidData/darcy".
            reduced_resolution (int): reduced spatial resolution, default: 1.
            reduced_batch (int): reduced batch, default: 1.
            split (str): dataset split which can be train, val or test, default: train.

        Returns:
            inputs, label, mask, case_params, self.grid, case_id
        shape:
            (x, y, c), (x, y, c), (x, y, 1), (x, y, 1), (x, y, 2), (1)
        """
        super().__init__()

        assert split in ["train", "val", "test"]

        self.multi_step_size = 1 # fix

        with h5py.File(os.path.join(saved_folder, filename), "r") as f:
            # compute dataset size
            nun_samples = f["tensor"].shape[0]
            train_size = int(0.8 * nun_samples)
            val_size = int(0.1 * nun_samples)

            # inputs and label
            if split == "train":
                inputs = f["nu"][:train_size:reduced_batch, ::reduced_resolution, ::reduced_resolution]
                label = f["tensor"][:train_size:reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
            elif split == "val":
                inputs = f["nu"][train_size:train_size+val_size:reduced_batch, ::reduced_resolution]
                label = f["tensor"][train_size:train_size+val_size:reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
            else:
                inputs = f["nu"][train_size+val_size::reduced_batch, ::reduced_resolution, ::reduced_resolution]
                label = f["tensor"][train_size+val_size::reduced_batch, :, ::reduced_resolution, ::reduced_resolution]

            # grid
            xcrd = f["x-coordinate"][::reduced_resolution]
            ycrd = f["y-coordinate"][::reduced_resolution]
            X, Y = np.meshgrid(xcrd, ycrd)
            self.grid = torch.from_numpy(np.stack([X, Y], axis=-1))
            
        self.inputs = torch.from_numpy(inputs).unsqueeze(-1) # [n, x, y, 1]
        self.label = torch.from_numpy(label).squeeze(1).unsqueeze(-1) # [n, x, y, 1]
        self.mask = torch.ones(self.inputs.shape[1:3]).unsqueeze(-1) # [x, y, 1]
        self.case_params = torch.full(self.inputs.shape[1:3], self.beta).unsqueeze(-1) # [x, y, num_case_params]

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.label[idx], self.mask, self.case_params, self.grid, idx