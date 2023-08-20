import os
import torch
import copy
import pandas as pd
import os.path as osp
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data.separate import separate


# Alters PygGraphPropPredDataset.get_idx_split() method to work with
# new defined splitting types
def get_idx_split(self, split_type=None):
    if split_type is None:
        split_type = self.meta_info['split']

    path = osp.join(self.root, 'split', split_type)

    # short-cut if split_dict.pt exists
    if os.path.isfile(os.path.join(path, 'split_dict.pt')):
        return torch.load(os.path.join(path, 'split_dict.pt'))

    if split_type == 'cluster':
        train_src_idx = pd.read_csv(osp.join(path, 'train_src.csv.gz'), compression='gzip', header=None).values.T[0]
        train_tar_idx = pd.read_csv(osp.join(path, 'train_tar.csv.gz'), compression='gzip', header=None).values.T[0]
        test_tar_idx = pd.read_csv(osp.join(path, 'test_tar.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train_src': torch.tensor(train_src_idx, dtype=torch.long),
                'train_tar': torch.tensor(train_tar_idx, dtype=torch.long),
                'test_tar': torch.tensor(test_tar_idx, dtype=torch.long)}
    elif split_type == 'species_adaptation':
        train_src_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        train_tar_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train_src': torch.tensor(train_src_idx, dtype=torch.long),
                'train_tar': torch.tensor(train_tar_idx, dtype=torch.long)}
    else:
        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype=torch.long),
                'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}


# Alters InMemoryDataset.get() method to return indexes of data
def get(self, idx: int) -> Data:
    if self.len() == 1:
        return copy.copy(self.data), idx

    if not hasattr(self, '_data_list') or self._data_list is None:
        self._data_list = self.len() * [None]
    elif self._data_list[idx] is not None:
        return copy.copy(self._data_list[idx]), idx

    data = separate(
        cls=self.data.__class__,
        batch=self.data,
        idx=idx,
        slice_dict=self.slices,
        decrement=False,
    )

    self._data_list[idx] = copy.copy(data)

    return data, idx


PygGraphPropPredDataset.get_idx_split = get_idx_split
InMemoryDataset.get = get
