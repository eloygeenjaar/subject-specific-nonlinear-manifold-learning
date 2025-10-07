import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from torch.utils.data import Dataset


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.loc[key, 'average']

    def result(self):
        return dict(self._data.average)

def generate_run_name(hyperparameter_dict):
    run_name = ''
    for hp_name, hp_val in hyperparameter_dict.items():
        if hp_name != 'seed':
            run_name += f'{hp_name}:{hp_val}-'
    run_name += f'seed:{hyperparameter_dict["seed"]}'
    return run_name

class SubsetDataset(Dataset):
    def __init__(self, dataset, subset_ratio):
        self.num_subjects = dataset.num_subjects
        self.old_timesteps = dataset.num_timesteps
        self.num_timesteps = int(self.old_timesteps * subset_ratio) 
        self.data = dataset.data.view(self.num_subjects, self.old_timesteps, -1)[:, :self.num_timesteps]
        self.data = torch.reshape(self.data, (self.num_subjects * self.num_timesteps, -1))
        print(f'New data timesteps: {self.num_timesteps}')
        self.subject_ixs = dataset.subject_ixs
        self.temporal_ixs = dataset.temporal_ixs[:self.num_timesteps]
        self.y = dataset.y

    def __len__(self):
        return self.num_subjects

    def __getitem__(self, ix):
        # Select the data for the subject
        x = self.data[ix]
        y = self.y[ix]
        subject_ix = self.subject_ixs[ix]
        temporal_ix = self.temporal_ixs
        return x, subject_ix, temporal_ix, y
