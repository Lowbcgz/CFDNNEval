import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class NSCHDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'ibc_phi_ca_mob_re_eps',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 stable_state_diff = 0.001,
                 norm_props = True,
                 reshape_parameters = True,
                 multi_step_size = 1,
                 ):
        
        '''
        Args:
            filename(str): the file name of dataset, such as "tube_dev.hdf5"
            saved_folder(str) : The path to the folder where the dataset is stored , "/home/data2/cfdbench_hdf5/tube/"
            case_name(str): Decide what type of dataset to use, for instance, "bc", "geo", "bc_geo" ...
            reduced_resolution(int): reduced spatial resolution, default:1
            reduced_batch(int): reduced batch, default:1 
            stable_state_diff(float): If the interval between two frames is less than this value, the following data is not taken, default:0.001
            norm_props(bool): if True, normalize the viscosity and density, default:True
            reshape_parameters(bool): if True, reshape the parameters, (cases, p) -> (cases, x, y, p)
        Returns:
            input, label, mask, case_params, self.grid, case_id
        shape:
            (x, y, c), (x, y, c), (x, y, 1), (x, y, p), (x, y, 2), (1)
        '''
        self.multi_step_size = multi_step_size
        self.case_name = case_name
        self.inputs = []
        self.labels = []
        self.physic_prop = []
        self.case_ids = []
        self.masks = []
        # cnt = 0 # for reduced batch

        fuvp_list=[]
        mobs_list=[]
        cas_list=[]
        res_list=[]
        eps_list=[]
        root_path = os.path.join(saved_folder, filename)
        with h5py.File(root_path, 'r') as f:
            # collect data
            for name in f.keys():
                if name in case_name.split('_'):
                    # print(f"{name} in {case_name}"  )
                    data_group = f[name]
                    fuvp_list.append(np.array(data_group["fuvp"],dtype=np.float32))
                    mobs_list.append(np.array(data_group["mobs"],dtype=np.float32))
                    cas_list.append(np.array(data_group["CAs"],dtype=np.float32))
                    res_list.append(np.array(data_group["Res"],dtype=np.float32))
                    eps_list.append(np.array(data_group["eps"],dtype=np.float32))
        fuvp=np.concatenate(fuvp_list,axis=0)[::reduced_batch]
        mobs=np.concatenate(mobs_list,axis=0)[::reduced_batch]
        cas=np.concatenate(cas_list,axis=0)[::reduced_batch]
        res=np.concatenate(res_list,axis=0)[::reduced_batch]
        eps=np.concatenate(eps_list,axis=0)[::reduced_batch]

        physic_prop= np.stack([cas, res, mobs, eps],axis=-1) # (B, 4)
        if norm_props:
            self.normalize_physics_props(physic_prop)
        # breakpoint()

        # print(fuvp.shape) # (B, T, Nx*Ny, 6)  6:(x,y,phi,u,v,pressure)
        fuvp= fuvp.reshape(fuvp.shape[0],fuvp.shape[1],66,66,6)
        fuv = fuvp[:,:,:,:, :5] # (B, T, Nx, Ny, 5)
        # idx = 0 # The index to record data corresponding to each frame
        fuv = fuv[:,:, ::reduced_resolution, ::reduced_resolution] # (B, T, Nx, Ny, 5)
        

        # filter the vaild frames
        for i in range(fuv.shape[0]):
            inputs= fuv[i, :-1]
            outputs = fuv[i, 1:]
            num_steps = len(inputs)
            for t in range(num_steps):
                if np.isnan(inputs[t]).any() or np.isnan(outputs[t]).any():
                    print(f"Invalid frame {t} in case {i}")
                    break
                inp_magn = np.sqrt(np.sum(inputs[t, :, :, 2:] ** 2, axis=-1))
                out_magn = np.sqrt(np.sum(outputs[t, :, :, 2:] ** 2, axis=-1))
                # out_magn = np.sqrt(outputs[t, :, :, 3] ** 2 + outputs[t, :, :, 4] ** 2)
                diff = np.abs(inp_magn - out_magn).mean()
                if diff < stable_state_diff:
                    print(f"Converged at {t} in case {i}")
                    break
                
                if t+1 >= multi_step_size:
                    self.inputs.append(torch.from_numpy(inputs[t+1-multi_step_size, :,:, 2:]).float())  
                    self.labels.append(torch.from_numpy(outputs[t+1-multi_step_size:t+1, :,:,2:]).float())
                    self.case_ids.append(i)

        #################################################
                        
        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, x, y, 3)
        self.labels = torch.stack(self.labels).float() #(Total frames, multi_step_size, x, y, 3)
        self.case_ids = np.array(self.case_ids) #(Total frames)

        self.masks = torch.ones_like(self.inputs[0,...,0:1]).float() #(x, y, 1)
        
        if self.multi_step_size==1:
            self.labels = self.labels.squeeze(1)
        
        _, x, y, _ = self.inputs.shape
        if reshape_parameters:
            #process the parameters shape
            self.physic_prop = torch.from_numpy(physic_prop).float() #(Total cases, 4)
            cases, p = self.physic_prop.shape
            
            self.physic_prop = self.physic_prop.reshape(cases, 1, 1, p)
            self.physic_prop = self.physic_prop.repeat(1, x, y, 1) #(cases, x, y, 4)
        else:
            self.physic_prop = torch.from_numpy(physic_prop).float() #(Total cases, 4)

        #get grid
        self.grid = torch.from_numpy(fuv[0,0,:,:,:2]).float()  # (x, y, 2)

        # print(f"shape of inputs: {self.inputs.shape}")
        

    def normalize_physics_props(self, physic_prop):
        """
        Normalize the physics properties in-place.
        """
        physic_prop = physic_prop/np.array([100.0,100.0,1.0,0.1])  # cas, res, mobs, eps
        
    def apply_norm(self, channel_min, channel_max):
        self.inputs = (self.inputs - channel_min) / (channel_max - channel_min)
        self.labels = (self.labels - channel_min) / (channel_max - channel_min)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]  # (x, y, 2)
        label = self.labels[idx]  # (multi_step_size, x, y, 2) or (x, y, 2)
        mask = self.masks # (x, y, 1)
        if self.multi_step_size > 1:
            mask = mask.unsqueeze(0).repeat(self.multi_step_size, 1, 1, 1)
        case_id = self.case_ids[idx]
        physic_prop = self.physic_prop[case_id] #(x, y, p)
        return input, label, mask, physic_prop, self.grid, case_id
