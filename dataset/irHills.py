import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset

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
            (nx, c), (nx, c), (nx, 1), (nx, p), (nx, 2), (1)
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

        cnt = 0 # for reduced batch

        root_path = os.path.join(saved_folder, filename)
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            idx = 0 # The index to record data corresponding to each frame
            for name in f.keys():
                if name in case_name.split('_'):
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
                        # load parameters
                        # read some parameters to pad and create mask, Remove some 
                        # parameters that are not used in training，and prepare for normalization 
                        this_case_params = {}
                        for param_name in data_keys:
                                if param_name in ['Vx', 'Vy', 'P', 'grid']:
                                    continue
                                this_case_params[param_name] = np.array(data[param_name], dtype=np.float32)[0]
                        
                        # if name == 'irRE' and (this_case_params['RE'] < 15 or this_case_params['RE'] > 99999):
                        #     continue
                        
                        #############################################################
                        #load u ,v, p, grid and get mask
                        u, v, w, p = np.array(data['Vx'], dtype=np.float32), np.array(data['Vy'], np.float32), np.array(data['Vz'], np.float32), np.array(data['P'], np.float32)
                        # print(u.shape, v.shape, p.shape)
                        u = u[::reduced_resolution].transpose(1, 0) # (T, nx)
                        v = v[::reduced_resolution].transpose(1, 0) # (T, nx)
                        w = w[::reduced_resolution].transpose(1, 0) # (T, nx)
                        p = p[::reduced_resolution].transpose(1, 0) # (T, nx)
                        #grid: [nx, 3]
                        grid = np.array(data['grid'][::reduced_resolution], np.float32)
                        grid[:,0] = grid[:,0] / self.statistics['pos_x_max']
                        grid[:,1] = grid[:,1] / self.statistics['pos_y_max']
                        grid[:,2] = grid[:,2] / self.statistics['pos_z_max']
                        self.grids.append(grid)
                        ### mask
                        mask = np.ones_like(u)
                        mask = torch.from_numpy(mask).float()
            
                        case_features = np.stack((u, v, w, p), axis=-1) # (T, nx, 4)
                        inputs = case_features[:-self.time_step_size]  # (T, nx, 4)
                        outputs = case_features[self.time_step_size:]  # (T, nx, 4)
                        assert len(inputs) == len(outputs)

                        num_steps = len(inputs)
                        # Loop frames, get input-output pairs
                        # Stop when converged
                        for i in range(num_steps):
                            if i+1 >= multi_step_size:
                                self.inputs.append(torch.from_numpy(inputs[i+1-multi_step_size]).float())  # (nx, 4)
                                self.labels.append(torch.from_numpy(outputs[i+1-multi_step_size:i+1]).float())  # (multi_step, nx, 4)
                                self.case_ids.append(idx)
                                #######################################################
                                #mask
                                #If each frame has a different mask, it needs to be rewritten 
                                self.masks.append(mask[i+1-multi_step_size:i+1, ...].unsqueeze(-1))
                        #norm props
                        if norm_props:
                            self.normalize_physics_props(this_case_params)

                        params_keys = ['RE']
                        case_params_vec = []
                        for k in params_keys:
                            case_params_vec.append(this_case_params[k])
                        case_params = torch.tensor(case_params_vec)  #(p)
                        self.case_params.append(case_params)
                        #################################################
                        idx += 1

        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, nx, 4)
        self.labels = torch.stack(self.labels).float() #(Total frames, multi_step, nx, 4)
        self.case_ids = np.array(self.case_ids) #(Total frames)
        self.masks = torch.stack(self.masks).float() #(Total frames, nx, 1)
        self.grids = torch.from_numpy(np.stack(self.grids)).float()

        if self.multi_step_size==1:
            self.labels = self.labels.squeeze(1)
            self.masks = self.masks.squeeze(1)

        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params).float() #(cases, p)
            cases, p = self.case_params.shape
            _, nx, _ = self.inputs.shape
            self.case_params = self.case_params.reshape(cases, 1, p)
            self.case_params = self.case_params.repeat(1, nx, 1) #(cases, nx, p)
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
            case_params["RE"] - 505.6250000000 
        ) / 299.4196166992

    def apply_norm(self, channel_min, channel_max):
        self.inputs = (self.inputs - channel_min) / (channel_max - channel_min)
        self.labels = (self.labels - channel_min) / (channel_max - channel_min)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]               # (nx, 4)
        label = self.labels[idx]                # (nx, 4)
        mask = self.masks[idx]                  # (nx, 1)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id] # (nx, p)
        grid = self.grids[case_id]              # (nx, 3)
        # print(inputs.shape, label.shape, mask.shape, case_params.shape, grid.shape, case_id)
        return inputs, label, mask, case_params, grid, case_id
