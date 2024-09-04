import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset

class HillsDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'rRE',
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
        self.case_params = []
        self.case_ids = []
        self.masks = []
        self.grids = []

        # perform normalization
        self.statistics = {}
        # -26.0151214600 66.4416732788 -31.5145797729 30.2989540100 -12.3325710297 25.9867649078 -310.9667663574 861.3582763672
        self.statistics['vel_x_min'] = -26.0151214600
        self.statistics['vel_x_max'] =  66.4416732788
        self.statistics['vel_y_min'] = -31.5145797729
        self.statistics['vel_y_max'] =  30.2989540100
        self.statistics['vel_z_min'] = -12.3325710297
        self.statistics['vel_z_max'] =  25.9867649078
        self.statistics['prs_min']   = -310.9667663574
        self.statistics['prs_max']   =  861.3582763672

        self.statistics['pos_x_min'] = 0.0   # left bound
        self.statistics['pos_x_max'] = 9.0   # right bound
        self.statistics['pos_y_min'] = 0.0   # lower bound
        self.statistics['pos_y_max'] = 4.5   # upper bound
        self.statistics['pos_z_min'] = 0.0   # lower bound
        self.statistics['pos_z_max'] = 3.035 # upper bound

        self.statistics['x_len'] = self.statistics['pos_x_max'] - self.statistics['pos_x_min']
        self.statistics['y_len'] = self.statistics['pos_y_max'] - self.statistics['pos_y_min']
        self.statistics['z_len'] = self.statistics['pos_z_max'] - self.statistics['pos_z_min']

        root_path = os.path.join(saved_folder, filename)
        with h5py.File(root_path, 'r') as f:
            case_id = 0 
            for name in f.keys():
                if name not in case_name.split('_'):
                    continue

                case_dataset = f[name]
                data_keys = sorted(case_dataset.keys())[::reduced_batch]
                for case in data_keys:
                    # load case data
                    data = case_dataset[case]

                    # load parameters
                    case_params = {}
                    for param_name in data.keys():
                        if param_name in ['Vx', 'Vy', 'Vz', 'P', 'grid']:
                            continue
                        case_params[param_name] = data[param_name][0]
                                        
                    # load u ,v, w, p
                    u = torch.from_numpy(data['Vx'][::reduced_resolution, ::reduced_resolution, ::reduced_resolution, ::self.time_step_size].transpose(3, 0, 1, 2)) # [T, x, y, z]
                    v = torch.from_numpy(data['Vy'][::reduced_resolution, ::reduced_resolution, ::reduced_resolution, ::self.time_step_size].transpose(3, 0, 1, 2))
                    w = torch.from_numpy(data['Vz'][::reduced_resolution, ::reduced_resolution, ::reduced_resolution, ::self.time_step_size].transpose(3, 0, 1, 2))
                    p = torch.from_numpy(data['P'][::reduced_resolution, ::reduced_resolution, ::reduced_resolution, ::self.time_step_size].transpose(3, 0, 1, 2))
                    # filter nan
                    index = torch.isnan(u)
                    u[index] = 0.
                    v[index] = 0.
                    w[index] = 0.
                    p[index] = 0.

                    # grid
                    grid = torch.from_numpy(data['grid'][::reduced_resolution, ::reduced_resolution, ::reduced_resolution])
                    # normalize the grid
                    grid[:, :, 0] = grid[:, :, 0] / self.statistics['x_len']
                    grid[:, :, 1] = grid[:, :, 1] / self.statistics['y_len']
                    grid[:, :, 2] = grid[:, :, 2] / self.statistics['z_len']
                    self.grids.append(grid) # [x, y, z, 3]
                    
                    # mask
                    mask = torch.ones_like(u) # [T, x, y, z]
                    mask[index] = 0.
                    mask.unsqueeze_(-1) # [T, x, y, z, 1]
                    
                    case_features = torch.stack((u, v, w, p), dim=-1) # (T, x, y, z, 4)
                    # Loop frames, get input-output pairs
                    for i in range(case_features.shape[0] - multi_step_size):
                        self.inputs.append(case_features[i])
                        self.labels.append(case_features[i+1:i+1+multi_step_size])
                        self.case_ids.append(case_id)
                        self.masks.append(mask[i+1:i+1+multi_step_size, ..., 0:1])

                    # norm case parameters
                    if norm_props:
                        self.normalize_physics_props(case_params)

                    params_keys = ['RE']
                    case_params_vec = []
                    for k in params_keys:
                        case_params_vec.append(case_params[k])
                    self.case_params.append(torch.tensor(case_params_vec, dtype=torch.float32))

                    case_id += 1

        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)
        self.masks = torch.stack(self.masks)
        self.case_ids = torch.tensor(self.case_ids)

        if self.multi_step_size == 1:
            self.labels.squeeze_(1)
            self.masks.squeeze_(1)

        # reshape parameters
        _, x, y, z, _ = self.inputs.shape
        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params) # (num_cases, p)
            cases, p = self.case_params.shape
            self.case_params = self.case_params.reshape(cases, 1, 1, 1, p)
            self.case_params = self.case_params.repeat(1, x, y, z, 1) #(cases, x, y, z, p)
        else:
            self.case_params = torch.stack(self.case_params)
        
        if num_samples_max > 0:
            assert num_samples_max < self.inputs.shape[0]
            self.inputs = self.inputs[:num_samples_max]
            self.labels = self.labels[:num_samples_max]
            self.case_ids = self.case_ids[:num_samples_max]
            self.masks = self.masks[:num_samples_max]

    def normalize_physics_props(self, case_params):
        """
        Normalize the physics properties in-place.
        """
        case_params["RE"] = (
            case_params["RE"] - 505.6250000000  
        ) / 299.4196166992

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        return self.inputs[idx], self.labels[idx], self.masks[idx], self.case_params[case_id], self.grids[case_id], case_id



class IRHillsDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'irRE',
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
        self.time_step_size = int(delta_time / data_delta_time)
        self.case_name = case_name
        self.multi_step_size = multi_step_size
        self.inputs = []
        self.labels = []
        self.masks = []
        self.case_params = []
        self.grids = []
        self.case_ids = []

        # dataset statistic
        self.statistics = {}
        self.statistics['vel_x_min'] = -28.0815334320
        self.statistics['vel_x_max'] =  66.4669570923
        self.statistics['vel_y_min'] = -32.2173538208
        self.statistics['vel_y_max'] =  31.7274246216
        self.statistics['vel_z_min'] = -12.8192892075
        self.statistics['vel_z_max'] =  26.2205390930
        self.statistics['prs_min']   = -313.4261779785
        self.statistics['prs_max']   =  862.0435180664

        self.statistics['pos_x_min'] = 0.0   # left bound
        self.statistics['pos_x_max'] = 9.0
        self.statistics['pos_y_min'] = 0.0   # lower bound
        self.statistics['pos_y_max'] = 4.5
        self.statistics['pos_z_min'] = 0.0   # lower bound
        self.statistics['pos_z_max'] = 3.035

        self.statistics['x_len'] = self.statistics['pos_x_max'] - self.statistics['pos_x_min']
        self.statistics['y_len'] = self.statistics['pos_y_max'] - self.statistics['pos_y_min']
        self.statistics['z_len'] = self.statistics['pos_z_max'] - self.statistics['pos_z_min']

        root_path = os.path.join(saved_folder, filename)
        with h5py.File(root_path, 'r') as f:
            case_id = 0
            for name in f.keys():
                if name not in case_name.split('_'):
                    continue
                
                case_dataset = f[name]
                data_keys = sorted(case_dataset.keys())[::reduced_batch]
                for case in data_keys:
                    # load case data
                    data = case_dataset[case]

                    # load parameters
                    case_params = {}
                    for param_name in data.keys():
                        if param_name in ['Vx', 'Vy', 'Vz', 'P', 'grid']:
                            continue
                        case_params[param_name] = data[param_name][0]

                    # load u, v, w, p
                    u = torch.from_numpy(data['Vx'][::reduced_resolution, ::self.time_step_size].transpose(1, 0)) # [T, nx]
                    v = torch.from_numpy(data['Vy'][::reduced_resolution, ::self.time_step_size].transpose(1, 0))
                    w = torch.from_numpy(data['Vz'][::reduced_resolution, ::self.time_step_size].transpose(1, 0))
                    p = torch.from_numpy(data['P'][::reduced_resolution, ::self.time_step_size].transpose(1, 0))

                    # grid
                    grid = data['grid'][::reduced_resolution]
                    # normalize the grid
                    grid[:,0] = grid[:,0] / self.statistics['pos_x_max']
                    grid[:,1] = grid[:,1] / self.statistics['pos_y_max']
                    grid[:,2] = grid[:,2] / self.statistics['pos_z_max']
                    self.grids.append(torch.from_numpy(grid)) # [num_points, 3]
                    
                    # mask
                    masks = torch.ones_like(u).unsqueeze(-1) # [T, num_points, 1]
                
                    case_features = torch.stack((u, v, w, p), dim=-1)
                    # Loop frames, get input-output pairs
                    for i in range(case_features.shape[0]-multi_step_size):
                        self.inputs.append(case_features[i])
                        self.labels.append(case_features[i+1:i+1+self.multi_step_size])
                        self.case_ids.append(case_id)
                        self.masks.append(masks[i+1:i+1+self.multi_step_size])

                    # normalize case parameters
                    if norm_props:
                        self.normalize_physics_props(case_params)

                    params_keys = ['RE']
                    case_params_vec = []
                    for k in params_keys:
                        case_params_vec.append(case_params[k])
                    self.case_params.append(torch.tensor(case_params_vec, dtype=torch.float32))

                    case_id += 1

        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)
        self.masks = torch.stack(self.masks)
        self.case_ids = torch.tensor(self.case_ids)

        if multi_step_size == 1:
            self.labels.squeeze_(1)
            self.masks.squeeze_(1)

        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params) # [cases, p]
            cases, p = self.case_params.shape
            _, nx, _ = self.inputs.shape
            self.case_params = self.case_params.reshape(cases, 1, p)
            self.case_params = self.case_params.repeat(1, nx, 1) # [cases, nx, p]
        else:
            self.case_params = torch.stack(self.case_params) # [cases, p]

        if num_samples_max > 0:
            assert num_samples_max < self.inputs.shape[0]
            self.inputs = self.inputs[:num_samples_max]
            self.labels = self.labels[:num_samples_max]
            self.case_ids = self.case_ids[:num_samples_max]
            self.masks = self.masks[:num_samples_max]
        
    def normalize_physics_props(self, case_params):
        """
        Normalize the physics properties in-place.
        """
        case_params["RE"] = (
            case_params["RE"] - 505.6250000000 
        ) / 299.4196166992

    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        return self.inputs[idx], self.labels[idx], self.masks[idx], self.case_params[case_id], self.grids[case_id], case_id