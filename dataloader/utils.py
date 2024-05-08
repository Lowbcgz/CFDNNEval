import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class TubeDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'bc_prop_geo',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 delta_time: float = 0.1,   
                 stable_state_diff = 0.001,
                 norm_props = True,
                 norm_bc = True,
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
            delta_time(float): Determine the spacing of each frame, default:0.1
            stable_state_diff(float): If the interval between two frames is less than this value, the following data is not taken, default:0.001
            norm_props(bool): if True, normalize the viscosity and density, default:True
            norm_bc(bool): if True, normalize the parameter of bc, default:True
            reshape_parameters(bool): if True, reshape the parameters, (cases, p) -> (cases, x, y, p)
            multi_step_size(int): Use the current frame to infer multi_step_size frames
        Returns:
            input, label, mask, case_params, self.grid, case_id
        shape:
            (x, y, c), (x, y, c), (x, y, 1), (x, y, p), (x, y, 2), (1)
        '''
        # The difference between input and output in number of frames.
        data_delta_time = 0.1
        self.time_step_size = int(delta_time / data_delta_time)
        self.case_name = case_name
        self.multi_step_size = multi_step_size
        self.inputs = []
        self.labels = []
        self.case_params = []
        self.case_ids = []
        self.masks = []

        cnt = 0 # for reduced batch

        root_path = os.path.abspath(saved_folder + filename)
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            idx = 0 # The index to record data corresponding to each frame
            for name in f.keys():
                if name in case_name:
                    data_group = f[name]
                    keys = list(data_group.keys())
                    keys.sort()
                    
                    for case in data_group.keys():
                        cnt += 1
                        if (cnt) % reduced_batch != 0:
                            continue
                        data = data_group[case]
                        ###################################################################
                        #load parameters
                        this_case_params = {}
                        for param_name in data.keys():
                                if param_name in ['Vx', 'Vy']:
                                    continue
                                this_case_params[param_name] = np.array(data[param_name], dtype=np.float32)[0]
                        # read some parameters to pad and create mask, Remove some 
                        # parameters that are not used in training，and normalization 
                           
                        if norm_props:
                            self.normalize_physics_props(this_case_params)
                        if norm_bc:
                            self.normalize_bc(this_case_params, "vel_in")
                               
                        keys = [
                            x for x in this_case_params.keys() if x not in ["rotated", "dx", "dy"]
                        ]
                        case_params_vec = []
                        for k in keys:
                            case_params_vec.append(this_case_params[k])
                        case_params = torch.tensor(case_params_vec)  #(p)
                        self.case_params.append(case_params)

                        #############################################################
                        #load u and v, and get mask
                        u, v = np.array(data['Vx'], dtype=np.float32), np.array(data['Vy'], np.float32)
                        u = u[:, ::reduced_resolution, ::reduced_resolution] # (T, x, y)
                        v = v[:, ::reduced_resolution, ::reduced_resolution] # (T, x, y)

                        ### mask
                        mask = np.ones_like(u)
                        #process the mask
                        # Pad the left side
                        mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
                        # # Pad the top and bottom
                        mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)

                        mask = torch.tensor(mask).float()

                        ################################################################
                        #pad u and v
                        # Pad the left side
                        u = np.pad(u, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=this_case_params["vel_in"])
                        v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
                        # # Pad the top and bottom
                        u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
                        v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
            
                        case_features = np.stack((u, v), axis=-1) # (T, x, y, 2)
                        inputs = case_features[:-self.time_step_size, :]  # (T, x, y, 2)
                        outputs = case_features[self.time_step_size:, :]  # (T, x, y, 2)
                        assert len(inputs) == len(outputs)

                        num_steps = len(inputs)
                        # Loop frames, get input-output pairs
                        # Stop when converged
                        for i in range(num_steps):
                            inp = torch.tensor(inputs[i], dtype=torch.float32)  # (x, y, 2)
                            out = torch.tensor(
                                outputs[i], dtype=torch.float32
                            )  # (x, y, 2)
                            # Check for convergence
                            inp_magn = torch.sqrt(inp[:,:,0] ** 2 + inp[:,:,1] ** 2)
                            out_magn = torch.sqrt(out[:,:,0] ** 2 + out[:,:,1] ** 2)
                            diff = torch.abs(inp_magn - out_magn).mean()
                            if diff < stable_state_diff:
                                print(
                                    f"Converged at {i} out of {num_steps},"
                                    f" {this_case_params}"
                                )
                                break
                            assert not torch.isnan(inp).any()
                            assert not torch.isnan(out).any()
                            if i+1 >= multi_step_size:
                                self.inputs.append(torch.tensor(inputs[i+1-multi_step_size], dtype=torch.float32))  # (x, y, 2)
                                self.labels.append(torch.tensor(outputs[i+1-multi_step_size:i+1], dtype=torch.float32))  # (multi_step, x, y, 2)
                                self.case_ids.append(idx)
                                #######################################################
                                #mask
                                #If each frame has a different mask, it needs to be rewritten 
                                self.masks.append(mask[i+1-multi_step_size:i+1, ...].unsqueeze(-1))

                        #################################################
                        idx += 1
                        
        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, x, y, 2)
        self.labels = torch.stack(self.labels).float() #(Total frames, x, y, 2)
        self.case_ids = np.array(self.case_ids) #(Total frames)
        self.masks = torch.stack(self.masks).float() #(Total frames, x, y, 2)
        if self.multi_step_size==1:
            self.labels = self.labels.squeeze(1)
            self.masks = self.masks.squeeze(1)

        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params).float() #(cases, p)
            cases, p = self.case_params.shape
            _, x, y, _ = self.inputs.shape
            self.case_params = self.case_params.reshape(cases, 1, 1, p)
            self.case_params = self.case_params.repeat(1, x, y, 1) #(cases, x, y, p)


        #get grid
        grid_x = torch.tensor(np.linspace(0, 1, x), dtype=torch.float)
        grid_x = grid_x.reshape(x, 1).repeat([1, y])
        grid_y = torch.tensor(np.linspace(0, 1, y), dtype=torch.float)
        grid_y = grid_y.reshape(1, y).repeat([x, 1])
        self.grid = torch.stack([grid_x, grid_y], axis = -1)  # (x, y, 2)


    def normalize_physics_props(self, case_params):
        """
        Normalize the physics properties in-place.
        """
        density_mean = 5
        density_std = 4
        viscosity_mean = 0.00238
        viscosity_std = 0.005
        case_params["density"] = (
            case_params["density"] - density_mean
        ) / density_std
        case_params["viscosity"] = (
            case_params["viscosity"] - viscosity_mean
        ) / viscosity_std

    def normalize_bc(self, case_params, key):
        """
        Normalize the boundary conditions in-place.
        """
        case_params[key] = case_params[key] / 50 - 0.5
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]  # (x, y, 2)
        label = self.labels[idx]  # (x, y, 2)
        mask = self.masks[idx] # (x, y, 1)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id] #(x, y, p)
        return input, label, mask, case_params, self.grid, case_id

class NSCH_Dataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'bc_ca_mob_phi_pre_re_uv0',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 stable_state_diff = 0.001,
                 norm_props = True,
                 reshape_parameters = True,
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
        root_path = os.path.abspath(saved_folder + filename)
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
        fuvp=np.concatenate(fuvp_list,axis=0)[::reduced_batch]
        mobs=np.concatenate(mobs_list,axis=0)[::reduced_batch]
        cas=np.concatenate(cas_list,axis=0)[::reduced_batch]
        res=np.concatenate(res_list,axis=0)[::reduced_batch]

        physic_prop= np.stack([cas, res, mobs],axis=-1) # (B, 3)
        if norm_props:
            self.normalize_physics_props(physic_prop)
        # breakpoint()

        # print(fuvp.shape) # (B, T, Nx*Ny, 6)  6:(x,y,phi,u,v,pressure)
        fuvp= fuvp.reshape(fuvp.shape[0],fuvp.shape[1],66,66,6)
        # idx = 0 # The index to record data corresponding to each frame
        fuvp = fuvp[:,:, ::reduced_resolution, ::reduced_resolution] # (B, T, Nx, Ny, 6)
        

        # filter the vaild frames
        for i in range(fuvp.shape[0]):
            inputs= fuvp[i, :-1]
            outputs = fuvp[i, 1:]
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
                self.inputs.append(torch.from_numpy(inputs[t, :, :, 2:]))
                self.labels.append(torch.from_numpy(outputs[t, :, :, 2:]))
                self.case_ids.append(i)
                
        #################################################
                        
        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, x, y, 4)
        self.labels = torch.stack(self.labels).float() #(Total frames, x, y, 4)
        self.case_ids = np.array(self.case_ids) #(Total frames)

        self.masks = torch.ones_like(self.inputs[0,...,0:1]).float() #(x, y, 1)

        if reshape_parameters:
            #process the parameters shape
            self.physic_prop = torch.from_numpy(physic_prop).float() #(Total cases, 3)
            cases, p = self.physic_prop.shape
            _, x, y, _ = self.inputs.shape
            self.physic_prop = self.physic_prop.reshape(cases, 1, 1, p)
            self.physic_prop = self.physic_prop.repeat(1, x, y, 1) #(cases, x, y, p)

        #get grid
        self.grid = torch.from_numpy(fuvp[0,0,:,:,:2]).float()  # (x, y, 2)

        # print(f"shape of inputs: {self.inputs.shape}")
        

    def normalize_physics_props(self, physic_prop):
        """
        Normalize the physics properties in-place.
        """
        physic_prop = physic_prop/np.array([100.0,100.0,1.0])
        
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]  # (x, y, 2)
        label = self.labels[idx]  # (x, y, 2)
        mask = self.masks # (x, y, 1)
        case_id = self.case_ids[idx]
        physic_prop = self.physic_prop[case_id] #(x, y, p)
        return input, label, mask, physic_prop, self.grid, case_id

class Darcy_Dataset(Dataset):
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

        root_path = os.path.abspath(saved_folder + filename)
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

class Cavity_Dataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'bc_prop_geo',
                 reduced_resolution = 1,
                 reduced_batch = 1,
                 data_delta_time = 0.1,
                 delta_time: float = 0.1,   
                 num_samples_max = -1,
                 stable_state_diff = 0.0001,
                 norm_props = True,
                 norm_bc = True,
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
            delta_time(float): Determine the spacing of each frame, default:0.1
            stable_state_diff(float): If the interval between two frames is less than this value, the following data is not taken, default:0.001
            norm_props(bool): if True, normalize the viscosity and density, default:True
            norm_bc(bool): if True, normalize the parameter of bc, default:True
            reshape_parameters(bool): if True, reshape the parameters, (cases, p) -> (cases, x, y, p)
            multi_step_size(int): Use the current frame to infer multi_step_size frames
        Returns:
            input, label, mask, case_params, self.grid, case_id
        shape:
            (x, y, c), (x, y, c), (x, y, 1), (x, y, p), (x, y, 2), (1)
        '''
        # The difference between input and output in number of frames.
        self.time_step_size = int(delta_time / data_delta_time)
        self.case_name = case_name
        self.multi_step_size = multi_step_size
        self.inputs = []
        self.labels = []
        self.case_params_dicts = []
        self.case_params = []
        self.case_ids = []
        self.masks = []
        self.grids = []

        cnt = 0 # for reduced batch

        root_path = os.path.abspath(saved_folder + filename)
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            idx = 0 # The index to record data corresponding to each frame
            for name in f.keys():
                if name in case_name:
                    data_group = f[name]
                    keys = list(data_group.keys())
                    keys.sort()
                    
                    for case in keys:
                        cnt += 1
                        if (cnt) % reduced_batch != 0:
                            continue
                        data = data_group[case]
                        data_keys = list(data.keys())
                        data_keys.sort()
                        ###################################################################
                        #load parameters
                        this_case_params = {}
                        for param_name in data_keys:
                                if param_name in ['Vx', 'Vy', 'P', 'grid']:
                                    continue
                                this_case_params[param_name] = np.array(data[param_name], dtype=np.float32)[0]
                        self.case_params_dicts.append(this_case_params)
                        # read some parameters to pad and create mask, Remove some 
                        # parameters that are not used in training，and prepare for normalization 
    
                        #############################################################
                        #load u ,v, p, grid and get mask
                        u, v, p = np.array(data['Vx'], dtype=np.float32), np.array(data['Vy'], np.float32), np.array(data['P'], np.float32)
                        u = u[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        v = v[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        p = p[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        #grid
                        grid = np.array(data['grid'], np.float32)
                        self.grids.append(grid)
                        ### mask
                        mask = np.ones_like(u)
                        mask = torch.tensor(mask).float()
            
                        case_features = np.stack((u, v, p), axis=-1) # (T, x, y, 3)
                        inputs = case_features[:-self.time_step_size]  # (T, x, y, 3)
                        outputs = case_features[self.time_step_size:]  # (T, x, y, 3)
                        assert len(inputs) == len(outputs)

                        num_steps = len(inputs)
                        # Loop frames, get input-output pairs
                        # Stop when converged
                        for i in range(num_steps):
                            inp = torch.tensor(inputs[i], dtype=torch.float32)  # (x, y, 3)
                            out = torch.tensor(
                                outputs[i], dtype=torch.float32
                            )  # (x, y, 3)
                            # Check for convergence
                            inp_magn = torch.sqrt(inp[:,:,0] ** 2 + inp[:,:,1] ** 2 + inp[:,:,2] ** 2)
                            out_magn = torch.sqrt(out[:,:,0] ** 2 + out[:,:,1] ** 2 + out[:,:,2] ** 2)
                            diff = torch.abs(inp_magn - out_magn).mean()
                            if diff < stable_state_diff:
                                print(
                                    f"Converged at {i} out of {num_steps},"
                                    f" {this_case_params}"
                                )
                                break
                            assert not torch.isnan(inp).any()
                            assert not torch.isnan(out).any()
                            if i+1 >= multi_step_size:
                                self.inputs.append(torch.tensor(inputs[i+1-multi_step_size], dtype=torch.float32))  # (x, y, 3)
                                self.labels.append(torch.tensor(outputs[i+1-multi_step_size:i+1], dtype=torch.float32))  # (multi_step, x, y, 3)
                                self.case_ids.append(idx)
                                #######################################################
                                #mask
                                #If each frame has a different mask, it needs to be rewritten 
                                self.masks.append(mask[i+1-multi_step_size:i+1, ...].unsqueeze(-1))

                        #################################################
                        idx += 1

        #normalize case parameters
        self.sum_information = {}
        self.Statistical_information = {}
        for case_params_dict in self.case_params_dicts:
            for u, v in case_params_dict.items():
                if u in self.sum_information:
                    self.sum_information[u] += v
                else:
                    self.sum_information[u] = v
        
        for u, v in self.sum_information.items():
            self.Statistical_information[u + '_mean'] = v / len(self.case_params_dicts)
            self.Statistical_information[u + '_std'] = 0
            for case_params_dict in self.case_params_dicts:
                self.Statistical_information[u + '_std'] += (case_params_dict[u] - self.Statistical_information[u + '_mean']) ** 2
            self.Statistical_information[u + '_std'] = np.sqrt(self.Statistical_information[u + '_std'] / len(self.case_params_dicts))
        
        for this_case_params in self.case_params_dicts:
            #normalization 
            if norm_props:
                self.normalize_physics_props(this_case_params)
            if norm_bc:
                self.normalize_bc(this_case_params, "vel_top")
                
            params_keys = [
                x for x in this_case_params.keys() if x not in ["rotated", "dx", "dy"]
            ]
            case_params_vec = []
            for k in params_keys:
                case_params_vec.append(this_case_params[k])
            case_params = torch.tensor(case_params_vec)  #(5)
            self.case_params.append(case_params)    

        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, x, y, 3)
        self.labels = torch.stack(self.labels).float() #(Total frames, x, y, 3)
        self.case_ids = np.array(self.case_ids) #(Total frames)
        self.masks = torch.stack(self.masks).float() #(Total frames, x, y, 1)
        self.grids = torch.tensor(np.hstack(self.grids)).float()

        if self.multi_step_size==1:
            self.labels = self.labels.squeeze(1)
            self.masks = self.masks.squeeze(1)

        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params).float() #(cases, p)
            cases, p = self.case_params.shape
            _, x, y, _ = self.inputs.shape
            self.case_params = self.case_params.reshape(cases, 1, 1, p)
            self.case_params = self.case_params.repeat(1, x, y, 1) #(cases, x, y, p)

        
        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.inputs.shape[0])
        else:
            num_samples_max = self.inputs.shape[0]
        
        self.inputs = self.inputs[:num_samples_max, ...]
        self.labels = self.labels[:num_samples_max, ...]
        self.case_ids = self.case_ids[:num_samples_max, ...]
        self.masks = self.masks[:num_samples_max, ...]

    def normalize_physics_props(self, case_params):
        """
        Normalize the physics properties in-place.
        """
        if self.Statistical_information['density_std'] != 0:
            case_params["density"] = (
                case_params["density"] - self.Statistical_information['density_mean']
            ) / self.Statistical_information['density_std']
        if self.Statistical_information['RE_std'] != 0:
            case_params["RE"] = (
                case_params["RE"] - self.Statistical_information['RE_mean']
            ) / self.Statistical_information['RE_std']

    def normalize_bc(self, case_params, key):
        """
        Normalize the boundary conditions in-place.
        """
        case_params[key] = (case_params[key] - self.Statistical_information[key + '_mean']) / self.Statistical_information[key + '_std']
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]  # (x, y, 2)
        label = self.labels[idx]  # (x, y, 2)
        mask = self.masks[idx] # (x, y, 1)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id] #(x, y, p)
        grid = self.grids[case_id] #(x, y, 2)
        return inputs, label, mask, case_params, grid, case_id


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    

###test
if __name__ == '__main__':
    flow_name = 'tube'
    data_type = 'train'
    filename = flow_name + '_' + data_type + '.hdf5'
    saved_folder = '/home/data2/cfdbench_hdf5/' + flow_name + '/'
    case_name = 'prop_bc_geo'
    data = TubeDataset(filename, saved_folder, data_delta_time=0.001, delta_time=0.01)
    inputs, label, mask, case_params, grid, case_id = data[0]
    print(inputs.shape, label.shape, mask.shape, case_params.shape, grid.shape)
    print(len(data))