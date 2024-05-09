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
