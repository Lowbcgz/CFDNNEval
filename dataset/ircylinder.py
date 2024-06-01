import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, IterableDataset

class IRCylinderDataset(Dataset):
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
        self.statistics['vel_x_mean'] = 22.5893936157
        self.statistics['vel_x_std']  = 25.1762752533
        self.statistics['vel_y_mean'] = 0.0292317439
        self.statistics['vel_y_std']  = 9.4673204422
        self.statistics['prs_mean']   = -189.3433685303
        self.statistics['prs_std']    = 596.6419677734

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
                        
                        if name == 'irRE' and (this_case_params['RE'] < 15 or this_case_params['RE'] > 99999):
                            continue
                        
                        #############################################################
                        #load u ,v, p, grid and get mask
                        u, v, p = np.array(data['Vx'], dtype=np.float32), np.array(data['Vy'], np.float32), np.array(data['P'], np.float32)

                        # print(u.shape, v.shape, p.shape)
                        u = u[::reduced_resolution].transpose(1, 0) # (T, nx)
                        v = v[::reduced_resolution].transpose(1, 0) # (T, nx)
                        p = p[::reduced_resolution].transpose(1, 0) # (T, nx)
                        #grid: [nx, 2]
                        grid = np.array(data['grid'][::reduced_resolution], np.float32)
                        grid[:,0] = grid[:,0] - self.statistics['pos_x_min']
                        grid[:,1] = grid[:,1] - self.statistics['pos_y_min']
                        self.grids.append(grid)
                        ### mask
                        mask = np.ones_like(u)
                        mask = torch.tensor(mask).float()
            
                        case_features = np.stack((u, v, p), axis=-1) # (T, nx, 3)
                        inputs = case_features[:-self.time_step_size]  # (T, nx, 3)
                        outputs = case_features[self.time_step_size:]  # (T, nx, 3)
                        assert len(inputs) == len(outputs)

                        num_steps = len(inputs)
                        # Loop frames, get input-output pairs
                        # Stop when converged
                        for i in range(num_steps):
                            if i+1 >= multi_step_size:
                                self.inputs.append(torch.tensor(inputs[i+1-multi_step_size], dtype=torch.float32))  # (nx, 3)
                                self.labels.append(torch.tensor(outputs[i+1-multi_step_size:i+1], dtype=torch.float32))  # (multi_step, nx, 3)
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
                        
                        # params_keys = [
                        #     x for x in this_case_params.keys() if x not in ["rotated", "dx", "dy"]
                        # ]
                        params_keys = ['RE', 'vel_top']
                        case_params_vec = []
                        for k in params_keys:
                            case_params_vec.append(this_case_params[k])
                        case_params = torch.tensor(case_params_vec)  #(p)
                        self.case_params.append(case_params)
                        #################################################
                        idx += 1

        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, nx, 3)
        self.labels = torch.stack(self.labels).float() #(Total frames, multi_step, nx, 3)
        self.case_ids = np.array(self.case_ids) #(Total frames)
        self.masks = torch.stack(self.masks).float() #(Total frames, nx, 1)
        self.grids = torch.tensor(np.stack(self.grids)).float()

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
        inputs = self.inputs[idx]               # (nx, 2)
        label = self.labels[idx]                # (nx, 2)
        mask = self.masks[idx]                  # (nx, 1)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id] # (nx, p)
        grid = self.grids[case_id]              # (nx, 2)
        return inputs, label, mask, case_params, grid, case_id



class IRCylinderDataset_NUNO(Dataset):
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
        self.statistics['vel_x_mean'] = 22.5893936157
        self.statistics['vel_x_std']  = 25.1762752533
        self.statistics['vel_y_mean'] = 0.0292317439
        self.statistics['vel_y_std']  = 9.4673204422
        self.statistics['prs_mean']   = -189.3433685303
        self.statistics['prs_std']    = 596.6419677734

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
                        
                        if name == 'irRE' and (this_case_params['RE'] < 15 or this_case_params['RE'] > 99999):
                            continue
                        
                        #############################################################
                        #load u ,v, p, grid and get mask
                        u, v, p = np.array(data['Vx'], dtype=np.float32), np.array(data['Vy'], np.float32), np.array(data['P'], np.float32)

                        # print(u.shape, v.shape, p.shape)
                        u = u[::reduced_resolution].transpose(1, 0) # (T, nx)
                        v = v[::reduced_resolution].transpose(1, 0) # (T, nx)
                        p = p[::reduced_resolution].transpose(1, 0) # (T, nx)
                        #grid: [nx, 2]
                        grid = np.array(data['grid'][::reduced_resolution], np.float32)
                        grid[:,0] = grid[:,0] - self.statistics['pos_x_min']
                        grid[:,1] = grid[:,1] - self.statistics['pos_y_min']
                        # self.grids.append(grid)
                        ### mask
                        mask = np.ones_like(u)
                        mask = torch.tensor(mask).float()
            
                        case_features = np.stack((u, v, p), axis=-1) # (T, nx, 3)


                        n_total = 1
                        n_subdomains = 6
                        oversamp_ratio = 1.5
                        input_xy = grid
                        T = case_features.shape[0]
                        input_u = case_features.reshape(1, T, -1, 3).transpose((0, 2, 1, 3))
                        point_cloud = grid.tolist()
                        # t1 = default_timer()
                        point_cloud = input_xy.tolist()
                        # Use kd-tree to generate subdomain division
                        tree= KDTree(
                            point_cloud, dim=2, n_subdomains=n_subdomains, 
                            n_blocks=8, return_indices=True
                        )
                        tree.solve()
                        # Gather subdomain info
                        bbox_sd = tree.get_subdomain_bounding_boxes()
                        indices_sd = tree.get_subdomain_indices()
                        input_xy_sd = np.zeros((np.max([len(indices_sd[i]) 
                            for i in range(n_subdomains)]), n_subdomains, 2))
                        for i in range(n_subdomains):
                            # Normalize to [-1, 1]
                            xy = input_xy[indices_sd[i], :]
                            _min, _max = np.min(xy, axis=0, keepdims=True), \
                                np.max(xy, axis=0, keepdims=True)
                            xy = (xy - _min) / (_max - _min) * 2 - 1
                            # Long side alignment
                            bbox = bbox_sd[i]
                            if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                                xy = np.flip(xy, axis=1)
                            input_xy_sd[:len(indices_sd[i]), i, :] = xy
                        # t2 = default_timer()
                        # print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

                        # if False:
                        #     input_u_sd_grid = np.load(PATH_U_SD_G)   
                        #         # shape: (1200, s1_padded, s2_padded, 31, 3, n_subdomains) 
                        #     input_u_sd = np.load(PATH_U_SD)          
                        #         # shape: (1200, n_points_sd_padded, 31, 3, n_subdomains) 
                        #     input_u_sd_mask = np.load(PATH_U_SD_M)   
                        #         # shape: (1, n_points_sd_padded, 1, 1, n_subdomains) 
                        # else:
                        # t1 = default_timer()
                        # print("Start interpolation...")
                        # Calculate the padded grid size
                        max_grid_size_x, max_grid_size_y = -1, -1
                        grid_sizes = []
                        is_transposed = [False] * n_subdomains
                        for i in range(n_subdomains):
                            n_points = len(indices_sd[i])
                            bbox = bbox_sd[i]
                            # Calculate the grid size, where the aspect ratio of the discrete grid 
                            # remains the same as the that of the original subdomain (bbox)
                            grid_size_x = np.sqrt(n_points * oversamp_ratio * \
                                (bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
                            grid_size_y = grid_size_x * (bbox[1][1] - bbox[1][0]) / (bbox[0][1] - bbox[0][0])
                            grid_size_x, grid_size_y = max(int(np.round(grid_size_x)), 2), \
                                max(int(np.round(grid_size_y)), 2)
                            grid_sizes.append((grid_size_x, grid_size_y))
                            # Long side alignment to reduce paddings
                            if bbox[0][1] - bbox[0][0] < bbox[1][1] - bbox[1][0]:
                                grid_size_x, grid_size_y = grid_size_y, grid_size_x
                                is_transposed[i] = True
                            max_grid_size_x, max_grid_size_y = max(max_grid_size_x, 
                                grid_size_x), max(max_grid_size_y, grid_size_y)

                        # Interpolation from point cloud to uniform grid
                        input_u_sd_grid = []
                        point_cloud = input_xy
                        point_cloud_val = np.transpose(input_u, (1, 2, 3, 0)) 
                        interp_linear = LinearNDInterpolator(point_cloud, point_cloud_val)
                        interp_rbf = RBFInterpolator(point_cloud, point_cloud_val, neighbors=6)
                        for i in range(n_subdomains):
                            grid_size_x, grid_size_y = grid_sizes[i]
                            bbox = bbox_sd[i]
                            # Linear interpolation
                            grid_x = np.linspace(bbox[0][0], bbox[0][1], num=grid_size_x)
                            grid_y = np.linspace(bbox[1][0], bbox[1][1], num=grid_size_y)
                            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
                            grid_val = interp_linear(grid_x, grid_y)
                            # Fill nan values
                            nan_indices = np.isnan(grid_val)[..., 0, 0, 0]
                            fill_vals = interp_rbf(np.stack((grid_x[nan_indices], grid_y[nan_indices]), axis=1))
                            grid_val[nan_indices] = fill_vals
                            # Resize to the same size via FFT-IFFT
                            freq = np.fft.rfft2(grid_val, axes=(0, 1))
                            s1_padded, s2_padded = max_grid_size_y, max_grid_size_x 
                            if is_transposed[i]:
                                s1_padded, s2_padded = s2_padded, s1_padded
                            square_freq = np.zeros((s1_padded, 
                                s2_padded // 2 + 1, T, 3, n_total)) + 0j
                            square_freq[:min(s1_padded//2, freq.shape[0]//2), 
                                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                                freq[:min(s1_padded//2, freq.shape[0]//2), 
                                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
                            square_freq[-min(s1_padded//2, freq.shape[0]//2):, 
                                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...] = \
                                freq[-min(s1_padded//2, freq.shape[0]//2):,
                                    :min(s2_padded//2+1, freq.shape[1]//2+1), ...]
                            grid_val = np.fft.irfft2(square_freq, 
                                s=(s1_padded, s2_padded), axes=(0, 1))
                            if is_transposed[i]:
                                grid_val = np.transpose(grid_val, (1, 0, 2, 3, 4))
                            input_u_sd_grid.append(np.transpose(grid_val, (4, 0, 1, 2, 3)))
                        input_u_sd_grid = np.transpose(np.array(input_u_sd_grid), (1, 2, 3, 4, 5, 0))

                        # Pad the point-cloud values of each subdomain to the same size
                        # Mask is used to ignore padded zeros when calculating errors
                        input_u_sd = np.zeros((n_total, 
                            np.max([len(indices_sd[i]) for i in range(n_subdomains)]), T, 3, n_subdomains))
                        input_u_sd_mask = np.zeros((1, 
                            np.max([len(indices_sd[i]) for i in range(n_subdomains)]), 1, 1, n_subdomains))
                        for i in range(n_subdomains):
                            input_u_sd[:, :len(indices_sd[i]), ..., i] = input_u[:, indices_sd[i], ...]
                            input_u_sd_mask[:, :len(indices_sd[i]), ..., i] = 1.
                        # print(input_u_sd.shape)
                            # if SAVE_PREP:
                            #     np.save(PATH_U_SD_G, input_u_sd_grid)
                            #     np.save(PATH_U_SD, input_u_sd) 
                            #     np.save(PATH_U_SD_M, input_u_sd_mask)
                            # t2 = default_timer()
                            # print("Finish interpolation, time elapsed: {:.1f}s".format(t2-t1))

                        input_xy_sd = torch.from_numpy(input_xy_sd).float()
                        # input_xy_sd = input_xy_sd.unsqueeze(0) # .repeat([batch_size, 1, 1, 1])\
                            # .permute(0, 2, 1, 3)\
                            # .reshape(batch_size * n_subdomains, -1, 1, 2)

                            # shape: (batch * n_subdomains, n_points_sd_padded, 1, 2)
                        self.input_xy_sd.append(input_xy_sd)

                        s1_padded, s2_padded = input_u_sd_grid.shape[1:3]
                        n_grid = s1_padded*s2_padded
                        input_u_sd_grid = input_u_sd_grid.reshape(s1_padded, s2_padded, T, -1).transpose((2, 0, 1, 3))
                        input_u_sd = input_u_sd.reshape(-1, T, 3, n_subdomains).transpose((1, 0, 3, 2))
                        input_u_sd_mask = torch.from_numpy(input_u_sd_mask.reshape(-1, 1, n_subdomains)).float()
                        grid = input_xy_sd
                        case_features = input_u_sd
                        mask = input_u_sd_mask.repeat([T, 1, 3, 1]).reshape(T, -1, 3,n_subdomains).transpose(0, 1, 3, 2)
                        # print(mask.shape, case_features.shape)
                        self.grids.append(grid)



                        inputs = case_features[:-self.time_step_size]  # (T, nx, 3)
                        outputs = case_features[self.time_step_size:]  # (T, nx, 3)
                        assert len(inputs) == len(outputs)

                        num_steps = len(inputs)
                        # Loop frames, get input-output pairs
                        # Stop when converged
                        for i in range(num_steps):
                            if i+1 >= multi_step_size:
                                self.inputs.append(torch.tensor(inputs[i+1-multi_step_size], dtype=torch.float32))  # (nx, 3)
                                self.labels.append(torch.tensor(outputs[i+1-multi_step_size:i+1], dtype=torch.float32))  # (multi_step, nx, 3)
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
                        
                        # params_keys = [
                        #     x for x in this_case_params.keys() if x not in ["rotated", "dx", "dy"]
                        # ]
                        params_keys = ['RE', 'vel_top']
                        case_params_vec = []
                        for k in params_keys:
                            case_params_vec.append(this_case_params[k])
                        case_params = torch.tensor(case_params_vec)  #(p)
                        self.case_params.append(case_params)
                        #################################################
                        idx += 1

        #Total frames = The sum of the number of frames for each case
        self.inputs = torch.stack(self.inputs).float() #(Total frames, nx, 3)
        self.labels = torch.stack(self.labels).float() #(Total frames, multi_step, nx, 3)
        self.case_ids = np.array(self.case_ids) #(Total frames)
        self.masks = torch.stack(self.masks).float() #(Total frames, nx, 1)
        self.grids = torch.tensor(np.stack(self.grids)).float()

        if self.multi_step_size==1:
            self.labels = self.labels.squeeze(1)
            self.masks = self.masks.squeeze(1)

        if reshape_parameters:
            #process the parameters shape
            self.case_params = torch.stack(self.case_params).float() #(cases, p)
            cases, p = self.case_params.shape
            _, s1, s2, _ = self.inputs.shape
            self.case_params = self.case_params.reshape(cases, 1, 1, p)
            self.case_params = self.case_params.repeat(1, s1, s2, 1) #(cases, nx, p)
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


        # data prepossing for NUNO
        # from ..model.NUNO.nufno2d import data_preprocessing
        # inputs_sd (N_total, max_Np, 8, C)
        # labels_sd (N_total, multi_step, max_Np, 8, C) or (N_total, max_Np, 8, C)
        # mask_sd (N_total, multi_step, max_Np, 8, 1) or (N_total, max_Np, 8, 1)
        # case_sd (N_total, max_Np, 8, p)
        # grid_sd (N_total, max_NP, 8, 2)
        # inputs_sd_grid (N_total,s1,s2, 8, C)

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
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]               # (nx, 2) 
        label = self.labels[idx]                # (nx, 2)
        mask = self.masks[idx]                  # (nx, 1)
        case_id = self.case_ids[idx]
        case_params = self.case_params[case_id] # (nx, p)
        grid = self.grids[case_id]
        aux_data = tensor(())              # (nx, 2)  
        return inputs, label, mask, case_params, grid, case_id, aux_data
        # return inputs_sd, labels_sd, mask_sd, cases_sd, grid_sd, case_id, inputs_sd_grid