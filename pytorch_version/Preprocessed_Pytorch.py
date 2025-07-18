
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class rawcvngenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), interaction_range=(4, 7)):
        self.samples = []
        self.input_shape = input_shape
        for fname in file_list:
            fpath = os.path.join(path, fname)
            
            try:
                with h5py.File(fpath, 'r') as f:
                    interaction = f['rec.training.trainingdata/interaction'][:, 0]
                    idxs = np.where((interaction >= interaction_range[0]) & (interaction <= interaction_range[1]))[0]
                    self.samples.extend([(fpath, idx) for idx in idxs])
            except OSError:
                continue



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, index = self.samples[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['rec.training.cvnmaps/cvnmap'][index].reshape(self.input_shape)
            y = f['rec.training.trainingdata/nuenergy'][index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)




"""class cvngenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), interaction_range=(4, 7)):
        self.samples = []
        self.input_shape = input_shape
        for fname in file_list:
            fpath = os.path.join(path, fname)
            print("Reading file:", fpath)
            try:
                with h5py.File(fpath, 'r') as f:
                    interaction = f['rec.training.trainingdata/interaction'][:, 0]
                    idxs = np.where((interaction >= interaction_range[0]) & (interaction <= interaction_range[1]))[0]
                    self.samples.extend([(fpath, idx) for idx in idxs])
            except OSError:
                continue"""


class cvngenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), mode='nue'):
        self.samples = []
        self.input_shape = input_shape
        self.mode = mode
        if mode == 'nue':
            ffix = 'event'
        elif mode == 'electron':
            ffix = 'prong'
        else:
            raise ValueError("Invalid mode. Choose 'nue' or 'electron'")

        for fname in file_list:
            fpath = os.path.join(path, fname)
            
            try:
                with h5py.File(fpath, 'r') as f:
                    n = len(f[ffix + 'trueE/df'])
                    self.samples.extend([(fpath, ffix, idx) for idx in range(n)])
            except Exception as e:
                print(f"❌ Failed to read {fpath}: {e}")

                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, ffix, index = self.samples[idx]
        with h5py.File(fpath, 'r') as f:
            image = f[ffix + 'map/df'][index].reshape(self.input_shape)
            target = f[ffix + 'trueE/df'][index]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)




class rawcvnpronggenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), key_prefix='rec.vtx.elastic.fuzzyk.png'):
        self.samples = []
        self.input_shape = input_shape
        for fname in file_list:
            fpath = os.path.join(path, fname)
            try:
                with h5py.File(fpath, 'r') as f:
                    if f[key_prefix + '.cvnmaps/cvnmap'].shape[0] > 0:
                        self.samples.extend([(fpath, i) for i in range(f[key_prefix + '.cvnmaps/cvnmap'].shape[0])])
            except OSError:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, index = self.samples[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['rec.vtx.elastic.fuzzyk.png.cvnmaps/cvnmap'][index].reshape(self.input_shape)
            y = f['rec.vtx.elastic.fuzzyk.png.truth/p.E'][index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class samplegenerator(Dataset):
    def __init__(self, sample_list, input_shape=(2, 100, 80)):
        self.samples = sample_list
        self.input_shape = input_shape

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, index = self.samples[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['rec.training.cvnmaps/cvnmap'][index].reshape(self.input_shape)
            y = f['rec.training.trainingdata/nuenergy'][index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class generator_mmap(Dataset):
    def __init__(self, feature_file, label_file):
        self.features = np.load(feature_file, mmap_mode='r')
        self.labels = np.load(label_file, mmap_mode='r')

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
