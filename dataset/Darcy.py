import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class PDEDarcyDataset(Dataset):

    beta = 1. # fix

    def __init__(self, 
                 filename,
                 saved_folder,
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 split = "train",
                 reshape_parameters = True,
                 ):
        """Dataset for Darcy flow in PDEBench.
        Args:
            filename (str): The file name of dataset file.
            saved_folder (str) : The path to the folder where the dataset is stored such as "/data1/FluidData/darcy".
            reduced_resolution (int): reduced spatial resolution, default: 1.
            reduced_batch (int): reduced batch, default: 1.
            split (str): Dataset split which can be train, val or test, default: train.

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
                inputs = f["nu"][train_size:train_size+val_size:reduced_batch, ::reduced_resolution, ::reduced_resolution]
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
        self.label = torch.from_numpy(label.transpose(0,2,3,1)) # [n, x, y, 1]
        self.mask = torch.ones(self.inputs.shape[1:3]).unsqueeze(-1) # [x, y, 1]
        if reshape_parameters:
            self.case_params = torch.full(self.inputs.shape[1:3], self.beta).unsqueeze(-1) # [x, y, num_case_params]
        else:
            self.case_params = torch.tensor(self.beta).unsqueeze(-1) # [num_case_params]

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.label[idx], self.mask, self.case_params, self.grid, idx