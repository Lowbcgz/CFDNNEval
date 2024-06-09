import torch
from torch.utils.data import Dataset
import h5py
import os
import numpy as np
class TGVDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'Re_ReD',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 num_samples_max = -1,
                 norm_props = True,
                 stable_state_diff = 0.0001,
                 reshape_parameters= True,
                 multi_step_size=1,
                 ):
        self.multi_step_size = multi_step_size
        self.case_name = case_name
        self.inputs = []
        self.labels = []
        self.physic_prop = []
        self.case_ids = []
        self.masks = []
        # cnt = 0 # for reduced batch

        uvp_list=[]
        re_list=[]
        V0_list=[]
        edge_list=[]
        nu_list=[]
        rho_list=[]
        root_path = os.path.abspath(saved_folder + filename)
        with h5py.File(root_path, 'r') as f:
            # collect data
            for name in f.keys():
                if name in case_name.split('_'):
                    # print(f"{name} in {case_name}"  )
                    data_group = f[name]
                    uvp_list.append(np.array(data_group["uvp"],dtype=np.float32))
                    re_list.append(np.array(data_group["Re"],dtype=np.float32))
                    
                    edge_list.append(np.array(data_group["edge"],dtype=np.float32))
                    nu_list.append(np.array(data_group["nu"],dtype=np.float32))
                    

        uvp=np.concatenate(uvp_list,axis=0)[::reduced_batch]
        re=np.concatenate(re_list,axis=0)[::reduced_batch]
        
        edge=np.concatenate(edge_list,axis=0)[::reduced_batch]
        nu=np.concatenate(nu_list,axis=0)[::reduced_batch]
        


        physic_prop= np.stack([re,edge,nu],axis=-1) # (B, 3)
        if norm_props:
            self.normalize_physics_props(physic_prop)

        # print(uvp.shape) # (B, T, Nx, Ny, 3)  3:(u,v,pressure)
        uvp = uvp[:,:, ::reduced_resolution, ::reduced_resolution] # (B, T, Nx, Ny, 3)
        

        # extract frames
        for i in range(uvp.shape[0]):
            inputs= uvp[i, :-1]
            outputs = uvp[i, 1:]
            num_steps = len(inputs)
            for t in range(num_steps):
                if np.isnan(inputs[t]).any() or np.isnan(outputs[t]).any():
                    print(f"Invalid frame {t} in case {i}")
                    break
                inp_magn = np.sqrt(np.sum(inputs[t] ** 2, axis=-1))
                out_magn = np.sqrt(np.sum(outputs[t] ** 2, axis=-1))
                
                diff = np.abs(inp_magn - out_magn).mean()
                if diff < stable_state_diff:
                    print(f"Converged at {t} in case {i}")
                    break
                
                if t+1 >= multi_step_size:
                    self.inputs.append(torch.from_numpy(inputs[t+1-multi_step_size]).float())  
                    self.labels.append(torch.from_numpy(outputs[t+1-multi_step_size:t+1]).float())
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
        #process the parameters shape
        if reshape_parameters:
            self.physic_prop = torch.from_numpy(physic_prop).float() #(Total cases, 3)
            cases, p = self.physic_prop.shape
            
            self.physic_prop = self.physic_prop.reshape(cases, 1, 1, p)
            self.physic_prop = self.physic_prop.repeat(1, x, y, 1) #(cases, x, y, p)
        else:
            self.physic_prop = torch.from_numpy(physic_prop).float()

        #get grid
        grid_x = torch.tensor(np.linspace(0, 1, x), dtype=torch.float)
        grid_x = grid_x.reshape(x, 1).repeat([1, y])
        grid_y = torch.tensor(np.linspace(0, 1, y), dtype=torch.float)
        grid_y = grid_y.reshape(1, y).repeat([x, 1])
        self.grid = torch.stack([grid_x, grid_y], axis = -1)  # (x, y, 2)

        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.inputs.shape[0])
        else:
            num_samples_max = self.inputs.shape[0]
        
        self.inputs = self.inputs[:num_samples_max, ...]
        self.labels = self.labels[:num_samples_max, ...]
        self.case_ids = self.case_ids[:num_samples_max, ...]
        # print(f"shape of inputs: {self.inputs.shape}")
        

    def normalize_physics_props(self, physic_prop):
        """
        Normalize the physics properties in-place.
        """
        physic_prop = physic_prop/np.array([1000.0,15.707,1.0])  #re,edge,nu
        
    def apply_norm(self, channel_min, channel_max):
        self.inputs = (self.inputs - channel_min) / (channel_max - channel_min)
        self.labels = (self.labels - channel_min) / (channel_max - channel_min)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]  # (x, y, 3)
        label = self.labels[idx]  # (x, y, 3)
        mask = self.masks # (x, y, 1)
        if self.multi_step_size>1:
            mask = mask.unsqueeze(0).repeat(self.multi_step_size, 1, 1, 1)
        case_id = self.case_ids[idx]
        physic_prop = self.physic_prop[case_id] #(x, y, p)
        return inputs, label, mask, physic_prop, self.grid, case_id
