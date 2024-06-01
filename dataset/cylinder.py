import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset

class CylinderDataset(Dataset):
    def __init__(self,
                 filename,
                 saved_folder='../data/',
                 case_name = 'rBC_rRE',
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
                 filter_outlier = True
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
        self.statistics['vel_x_mean'] = 26.0349330902
        self.statistics['vel_x_std']  = 24.1961174011
        self.statistics['vel_y_mean'] = 0.0479271673
        self.statistics['vel_y_std']  = 5.5985617638
        self.statistics['prs_mean']   = -55.1878814697
        self.statistics['prs_std']    = 208.3219757080

        self.statistics['pos_x_min'] = 1.875  # left bound
        self.statistics['pos_x_max'] = 2.875  # right bound
        self.statistics['pos_y_min'] = 3.5    # lower bound
        self.statistics['pos_y_max'] = 4.5    # upper bound

        self.statistics['x_len'] = self.statistics['pos_x_max'] - self.statistics['pos_x_min']
        self.statistics['y_len'] = self.statistics['pos_y_max'] - self.statistics['pos_y_min']

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
                        # read some parameters to pad and create mask, Remove some 
                        # parameters that are not used in training，and prepare for normalization 
                        this_case_params = {}
                        for param_name in data_keys:
                                if param_name in ['Vx', 'Vy', 'P', 'grid']:
                                    continue
                                this_case_params[param_name] = np.array(data[param_name], dtype=np.float32)[0]
                        
                        if name == 'rRE' and (this_case_params['RE'] < 15 or this_case_params['RE'] > 99999) and filter_outlier:
                            continue
                        
                        #############################################################
                        #load u ,v, p, grid and get mask
                        u, v, p = np.array(data['Vx'], dtype=np.float32), np.array(data['Vy'], np.float32), np.array(data['P'], np.float32)
                        index = np.isnan(u)
                        
                        u[np.isnan(u)] = 0
                        v[np.isnan(v)] = 0
                        p[np.isnan(p)] = 0
                        u = u[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        v = v[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        p = p[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        #grid: [x, y, 2]
                        grid = np.array(data['grid'][::reduced_resolution, ::reduced_resolution], np.float32)
                        grid[:,:,0] = grid[:,:,0] - self.statistics['pos_x_min']
                        grid[:,:,1] = grid[:,:,1] - self.statistics['pos_y_min']
                        self.grids.append(grid)
                        ### mask
                        index = index[::reduced_resolution, ::reduced_resolution].transpose(2, 0, 1) # (T, x, y)
                        mask = np.ones_like(u)
                        mask[index] = 0
                        mask = torch.from_numpy(mask)
            
                        case_features = np.stack((u, v, p), axis=-1) # (T, x, y, 3)
                        inputs = case_features[:-self.time_step_size]  # (T, x, y, 3)
                        outputs = case_features[self.time_step_size:]  # (T, x, y, 3)
                        assert len(inputs) == len(outputs)

                        num_steps = len(inputs)
                        # Loop frames, get input-output pairs
                        for i in range(num_steps):
                            if i+1 >= multi_step_size:
                                self.inputs.append(torch.from_numpy(inputs[i+1-multi_step_size]).float())  # (x, y, 3)
                                self.labels.append(torch.from_numpy(outputs[i+1-multi_step_size:i+1]).float())  # (multi_step, x, y, 3)
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
                        
                        params_keys = ['RE', 'vel_top']
                        case_params_vec = []
                        for k in params_keys:
                            case_params_vec.append(this_case_params[k])
                        case_params = torch.tensor(case_params_vec)  #(p)
                        self.case_params.append(case_params)
                        #################################################
                        idx += 1

        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, x, y, 3)
        self.labels = torch.stack(self.labels).float() #(Total frames, ms, x, y, 3)
        self.case_ids = np.array(self.case_ids) #(Total frames)
        self.masks = torch.stack(self.masks).float() #(Total frames, x, y, 1)
        self.grids = torch.from_numpy(np.stack(self.grids)).float()
        
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
        
        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.inputs.shape[0])
        else:
            num_samples_max = self.inputs.shape[0]
        
        self.inputs = self.inputs[:num_samples_max, ...]
        self.labels = self.labels[:num_samples_max, ...]
        self.case_ids = self.case_ids[:num_samples_max, ...]
        self.masks = self.masks[:num_samples_max, ...]

        # print(self.inputs.shape, self.labels.shape, self.case_ids.shape, self.masks.shape, self.case_params.shape)

    def normalize_physics_props(self, case_params):
        """
        Normalize the physics properties in-place.
        """
        case_params["RE"] = (
            case_params["RE"] - 8892.857   
        ) / 20801.535

    def normalize_bc(self, case_params):
        """
        Normalize the boundary conditions in-place.
        """
        case_params['vel_top'] = (case_params['vel_top'] - 18.366547) / 16.31889
    
    def apply_norm(self, channel_min, channel_max):
        self.inputs = (self.inputs - channel_min) / (channel_max - channel_min)
        self.labels = (self.labels - channel_min) / (channel_max - channel_min)

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
