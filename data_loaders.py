import numpy as np
import os
import pickle
import gzip
import requests
import glob
import h5py
import random
import math as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import copy
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_pde(root, batch_size, num_workers=32, subset=None, size="base"):
    
    train_data = MixedDataset(root,subset,if_test="train", ratio=1, size=size)
    
    if subset is None:
        subset = "cavity_ReD"
    val_data = MixedDataset(root, subset, if_test="val", ratio=1, size=size)
    test_data = MixedDataset(root, subset, if_test="test", ratio=1, size=size)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                num_workers=num_workers, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                num_workers=num_workers, shuffle=False)
    return train_loader, val_loader, test_loader


class MixedDataset(Dataset):
    def __init__(self, saved_folder, subset=None,
                 if_test="train", ratio=1,
                 size="base"):
        """
        
        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """
        self.size = size
        self.if_test = if_test
        # Define path to files
        if self.if_test == 'test' or self.if_test == 'val':
            self.root_path = os.path.abspath(saved_folder + f"/mixed_data_{if_test}/" + subset)
            p = self.root_path +"/shape.npy"
            s = np.load(p)[:,:1]
            s[:,0] -= 1
            if ratio < 1:
                res = s.shape[0] * ratio
                s = s[:int(res)]
            total = s.sum()
            self.shape_list = s
            stats = torch.from_numpy(np.load(self.root_path +"/stats.npy")).float()
            self.mean, self.std = stats[0], stats[1]
            self.grid = torch.from_numpy(np.load(self.root_path +"/grid.npy")).float()
            self.mask = torch.from_numpy(np.load(self.root_path +"/mask.npy")).float()
        elif self.if_test == 'train':
            self.root_path = os.path.abspath(saved_folder + "/mixed_data_train")
            train_names = ["cavity_bc", "cavity_re", "cavity_ReD", "cylinder_rBC", "cylinder_rRE", "nsch_ca", "nsch_eps", "nsch_mob", "nsch_re", "tgv_Re", "tgv_ReD", "tube_bc", "tube_geo", "tube_prop"]
            # ['cavity_re', 'cavity_bc', 'cavity_ReD', 'cylinder_rBC', 'cylinder_rRE', 'nsch_ca', 'nsch_mob', 'nsch_re', 'tgv_all', 'tube_bc', 'tube_geo', 'tube_prop']
            # load from subset
            # if subset is not None:
            #     train_names = [subset]
            self.dataset_num = len(train_names)
            self.train_name_dict = {k: train_names[k] for k in range(len(train_names))}
            self.index_dict = {}
            self.shape_list = []
            total = 0
            for i, name in enumerate(train_names):
                p = self.root_path + "/" + name +"/shape.npy"
                s = np.load(p)[:,:1]
                res = s.shape[0] * ratio
                s = s[:int(res)]
                s[:,0] -= 1
                self.shape_list.append( s.reshape(-1) )
                total += s.sum()
        else:
            raise ValueError("if_test must be 'train', 'val' or 'test'")
        self.total = total


    def __len__(self):
        return self.total

    def get_idx(self, idx):
        sidx = 0
        didx = 0
        tidx = 0
        for k in range(self.dataset_num):
            s = self.shape_list[k]
            if idx < s.sum():
                for i in range(len(s)):
                    if idx < s[i]:
                        return sidx, didx, idx
                    idx -= s[i]
                    didx += 1
            idx -= s.sum()
            sidx += 1
        return sidx, didx, tidx

    
    def __getitem__(self, idx):
        if self.if_test == 'test':
            didx = 0
            tidx = idx
            s = self.shape_list
            for i in range(len(s)):
                if tidx < s[i]:
                    break
                tidx -= s[i]
                didx += 1
            # print(self.root_path, didx, tidx)
            e = torch.from_numpy(np.load(self.root_path + "/" + str(didx) + "_" + self.size + "_embeddings.npy")).float()
            x = torch.from_numpy(np.load(self.root_path + "/" + str(didx) +".npy")[tidx]).float()
            y = torch.from_numpy(np.load(self.root_path + "/" + str(didx) +".npy")[tidx + 1]).float()
            x = x.reshape(4, 64, 64)
            y = y.reshape(4, 64, 64)
            x = torch.cat([x, self.grid])
            return (x, e), (y, self.mask), didx
        elif self.if_test == 'val':
            didx = 0
            tidx = idx
            s = self.shape_list
            for i in range(len(s)):
                if tidx < s[i]:
                    break
                tidx -= s[i]
                didx += 1
            # print(self.root_path, didx, tidx)
            e = torch.from_numpy(np.load(self.root_path + "/" + str(didx) + "_" + self.size + "_embeddings.npy")).float()
            x = torch.from_numpy(np.load(self.root_path + "/" + str(didx) +".npy")[tidx]).float()
            y = torch.from_numpy(np.load(self.root_path + "/" + str(didx) +".npy")[tidx + 1]).float()
            x = x.reshape(4, 64, 64)
            y = y.reshape(4, 64, 64)
            x = torch.cat([x, self.grid])
            return (x, e), (y, self.mask)
        elif self.if_test == 'train':
            sidx, didx, tidx = self.get_idx(idx)

            dsname = self.train_name_dict[sidx]
            p = self.root_path + "/" + dsname
            x = torch.from_numpy(np.load(p+ "/" + str(didx) +".npy")[tidx]).float()
            y = torch.from_numpy(np.load(p+ "/" + str(didx) +".npy")[tidx+1]).float()
            x = x.reshape(4, 64, 64)
            y = y.reshape(4, 64, 64)
            g = torch.from_numpy(np.load(p+ "/grid.npy")).float()
            e = torch.from_numpy(np.load(p+ "/" + str(didx) + "_" + self.size +"_embeddings.npy")).float()
            m = torch.from_numpy(np.load(p+ "/mask.npy")).float()
            x = torch.cat([x, g])
            return (x, e), (y, m)
        else:
            raise ValueError("if_test must be 'train', 'val' or 'test'")


if __name__ == '__main__':
    load_pde("datasets", 32, dataset='Burgers', reduced_resolution=1, prev_t=1, valid_split=-1, num_workers=0, subset=None, size="base")
