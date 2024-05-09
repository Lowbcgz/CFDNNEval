import os
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class CavityDataset(Dataset):
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