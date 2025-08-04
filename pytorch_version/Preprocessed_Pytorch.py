import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

def get_weights(y):
    w = np.zeros_like(y)
    w += np.where(np.logical_and(0<=y, y<0.5), np.ones_like(y), np.zeros_like(y))*42.5586924
    w += np.where(np.logical_and(0.5<=y, y<5), np.ones_like(y), np.zeros_like(y))*(83.0923-97.7887*y+35.1566*y**2-3.42726*y**3)
    w += np.where(np.logical_and(5<=y, y<15), np.ones_like(y), np.zeros_like(y))*(-137.471+68.8724*y-7.69719*y**2+0.26117*y**3)
    w += np.where(15<=y, np.ones_like(y), np.zeros_like(y))*45.196
    return w.reshape(-1)

class rawcvngenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), interaction_range=(4, 7), weighted=False):
        self.samples = []
        self.input_shape = input_shape
        self.weighted = weighted
        for fname in file_list:
            fpath = os.path.join(path, fname)
            
            try:
                with h5py.File(fpath, 'r') as f:
                    interaction = f['rec.training.trainingdata/interaction'][:, 0]
                    idxs = np.where((interaction >= interaction_range[0]) & (interaction <= interaction_range[1]))[0]
                    if self.weighted:
                        y = f['rec.training.trainingdata/nuenergy'][idxs]
                        weights = get_weights(y)
                        self.samples.extend([(fpath, idx, weight) for idx, weight in zip(idxs, weights)])
                    else:
                        self.samples.extend([(fpath, idx) for idx in idxs])
            except OSError:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.weighted:
            fpath, index, weight = self.samples[idx]
            with h5py.File(fpath, 'r') as f:
                x = f['rec.training.cvnmaps/cvnmap'][index].reshape(self.input_shape)
                y = f['rec.training.trainingdata/nuenergy'][index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)
        else:
            fpath, index = self.samples[idx]
            with h5py.File(fpath, 'r') as f:
                x = f['rec.training.cvnmaps/cvnmap'][index].reshape(self.input_shape)
                y = f['rec.training.trainingdata/nuenergy'][index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class cvngenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), mode='nue', weighted=False):
        self.samples = []
        self.input_shape = input_shape
        self.mode = mode
        self.weighted = weighted
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
                    if self.weighted:
                        y = f[ffix + 'trueE/df'][:]
                        weights = get_weights(y)
                        self.samples.extend([(fpath, ffix, idx, weight) for idx, weight in zip(range(n), weights)])
                    else:
                        self.samples.extend([(fpath, ffix, idx) for idx in range(n)])
            except Exception as e:
                print(f"❌ Failed to read {fpath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.weighted:
            fpath, ffix, index, weight = self.samples[idx]
            with h5py.File(fpath, 'r') as f:
                image = f[ffix + 'map/df'][index].reshape(self.input_shape)
                target = f[ffix + 'trueE/df'][index]
            return torch.tensor(image, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)
        else:
            fpath, ffix, index = self.samples[idx]
            with h5py.File(fpath, 'r') as f:
                image = f[ffix + 'map/df'][index].reshape(self.input_shape)
                target = f[ffix + 'trueE/df'][index]
            return torch.tensor(image, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class rawcvnpronggenerator(Dataset):
    def __init__(self, path, file_list, input_shape=(2, 100, 80), key_prefix='rec.vtx.elastic.fuzzyk.png', weighted=False):
        self.samples = []
        self.input_shape = input_shape
        self.weighted = weighted
        for fname in file_list:
            fpath = os.path.join(path, fname)
            try:
                with h5py.File(fpath, 'r') as f:
                    if f[key_prefix + '.cvnmaps/cvnmap'].shape[0] > 0:
                        if self.weighted:
                            y = f[key_prefix + '.truth/p.E'][:]
                            weights = get_weights(y)
                            self.samples.extend([(fpath, i, weight) for i, weight in zip(range(f[key_prefix + '.cvnmaps/cvnmap'].shape[0]), weights)])
                        else:
                            self.samples.extend([(fpath, i) for i in range(f[key_prefix + '.cvnmaps/cvnmap'].shape[0])])
            except OSError:
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.weighted:
            fpath, index, weight = self.samples[idx]
            with h5py.File(fpath, 'r') as f:
                x = f['rec.vtx.elastic.fuzzyk.png.cvnmaps/cvnmap'][index].reshape(self.input_shape)
                y = f['rec.vtx.elastic.fuzzyk.png.truth/p.E'][index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(weight, dtype=torch.float32)
        else:
            fpath, index = self.samples[idx]
            with h5py.File(fpath, 'r') as f:
                x = f['rec.vtx.elastic.fuzzyk.png.cvnmaps/cvnmap'][index].reshape(self.input_shape)
                y = f['rec.vtx.elastic.fuzzyk.png.truth/p.E'][index]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class samplegenerator(Dataset):
    def __init__(self, sample_list, input_shape=(2, 100, 80), weighted=False):
        self.samples = sample_list
        self.input_shape = input_shape
        self.weighted = weighted
        if weighted:
            ys = []
            for fpath, index in sample_list:
                with h5py.File(fpath, 'r') as f:
                    ys.append(f['rec.training.trainingdata/nuenergy'][index])
            self.weights = get_weights(np.array(ys))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, index = self.samples[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['rec.training.cvnmaps/cvnmap'][index].reshape(self.input_shape)
            y = f['rec.training.trainingdata/nuenergy'][index]
        if self.weighted:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(self.weights[idx], dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class generator_mmap(Dataset):
    def __init__(self, feature_file, label_file, weighted=False):
        self.features = np.load(feature_file, mmap_mode='r')
        self.labels = np.load(label_file, mmap_mode='r')
        self.weighted = weighted
        if weighted:
            self.weights = get_weights(self.labels)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.weighted:
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32), torch.tensor(self.weights[idx], dtype=torch.float32)
        else:
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)