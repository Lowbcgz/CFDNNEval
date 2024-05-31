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

        root_path = os.path.join(saved_folder, filename)
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
                        # read some parameters to pad and create mask, Remove some 
                        # parameters that are not used in training，and normalization 
                        this_case_params = {}
                        for param_name in data.keys():
                                if param_name in ['Vx', 'Vy']:
                                    continue
                                this_case_params[param_name] = np.array(data[param_name], dtype=np.float32)[0]
                        
                           
                        

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
                        #Dimensionless normalization
                        u = u / this_case_params['vel_in']
                        v = v / this_case_params['vel_in']
            
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

                        #norm props
                        if norm_props:
                            self.normalize_physics_props(this_case_params)
                        if norm_bc:
                            self.normalize_bc(this_case_params)
                               
                        keys = [
                            x for x in this_case_params.keys() if x not in ["rotated", "dx", "dy"]
                        ]
                        case_params_vec = []
                        for k in keys:
                            case_params_vec.append(this_case_params[k])
                        case_params = torch.tensor(case_params_vec)  #(p)
                        self.case_params.append(case_params)
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

        _, x, y, _ = self.inputs.shape
        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params).float() #(cases, p)
            cases, p = self.case_params.shape
            self.case_params = self.case_params.reshape(cases, 1, 1, p)
            self.case_params = self.case_params.repeat(1, x, y, 1) #(cases, x, y, p)
        else:
            self.case_params = torch.stack(self.case_params).float()

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
        density_mean = 45.43571472167969
        density_std = 47.47755432128906
        viscosity_mean = 0.32749998569488525
        viscosity_std = 0.31651002168655396
        case_params["density"] = (
            case_params["density"] - density_mean
        ) / density_std
        case_params["viscosity"] = (
            case_params["viscosity"] - viscosity_mean
        ) / viscosity_std

    def normalize_bc(self, case_params):
        """
        Normalize the boundary conditions in-place.
        """
        case_params['vel_in'] = (case_params['vel_in'] - 1.4371428489685059) / 1.0663825273513794
    
    def apply_norm(self, channel_min, channel_max):
        self.inputs = (self.inputs - channel_min) / (channel_max - channel_min)
        self.labels = (self.labels - channel_min) / (channel_max - channel_min)
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]  # (x, y, 2)
        label = self.labels[idx]  # (x, y, 2)
        mask = self.masks[idx] # (x, y, 1)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id] #(x, y, p)
        return input, label, mask, case_params, self.grid, case_id
