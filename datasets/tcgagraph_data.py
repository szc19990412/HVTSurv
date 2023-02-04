import random
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import torch.utils.data as data
from torch.utils.data import dataloader
from loguru import logger
logger.add('test.log')
logger

#---->BatchWSI
import torch_geometric
from typing import List

import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
import torch_geometric
from torch_geometric.data import Data

class BatchWSI(torch_geometric.data.Batch):
    def __init__(self):
        super(BatchWSI, self).__init__()
        pass
    
    @classmethod 
    def from_data_list(cls, data_list, follow_batch=[], exclude_keys=[], update_cat_dims={}):
            r"""Constructs a batch object from a python list holding
            :class:`torch_geometric.data.Data` objects.
            The assignment vector :obj:`batch` is created on the fly.
            Additionally, creates assignment batch vectors for each key in
            :obj:`follow_batch`.
            Will exclude any keys given in :obj:`exclude_keys`."""

            keys = list(set(data_list[0].keys) - set(exclude_keys))
            assert 'batch' not in keys and 'ptr' not in keys

            batch = cls() #BatchWSI
            for key in data_list[0].__dict__.keys():
                if key[:2] != '__' and key[-2:] != '__':
                    batch[key] = None

            batch.__num_graphs__ = len(data_list)
            batch.__data_class__ = data_list[0].__class__
            for key in keys + ['batch']:
                batch[key] = []
            batch['ptr'] = [0]
            cat_dims = {}
            device = None
            slices = {key: [0] for key in keys}
            cumsum = {key: [0] for key in keys}
            num_nodes_list = []
            for i, data in enumerate(data_list):
                for key in keys:
                    item = data[key]

                    # Increase values by `cumsum` value.
                    cum = cumsum[key][-1]
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    # Gather the size of the `cat` dimension.
                    size = 1
                    
                    if key in update_cat_dims.keys():
                        cat_dim = update_cat_dims[key]
                    else:
                        cat_dim = data.__cat_dim__(key, data[key])
                        # 0-dimensional tensors have no dimension along which to
                        # concatenate, so we set `cat_dim` to `None`.
                        if isinstance(item, Tensor) and item.dim() == 0:
                            cat_dim = None
                    
                    cat_dims[key] = cat_dim

                    # Add a batch dimension to items whose `cat_dim` is `None`:
                    if isinstance(item, Tensor) and cat_dim is None:
                        cat_dim = 0  # Concatenate along this new batch dimension.
                        item = item.unsqueeze(0)
                        device = item.device
                    elif isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device()

                    batch[key].append(item)  # Append item to the attribute list.

                    slices[key].append(size + slices[key][-1])
                    inc = data.__inc__(key, item)
                    if isinstance(inc, (tuple, list)):
                        inc = torch.tensor(inc)
                    cumsum[key].append(inc + cumsum[key][-1])

                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full((size, ), i, dtype=torch.long,
                                               device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))

                if hasattr(data, '__num_nodes__'):
                    num_nodes_list.append(data.__num_nodes__)
                else:
                    num_nodes_list.append(None)

                num_nodes = data.num_nodes
                if num_nodes is not None:
                    item = torch.full((num_nodes, ), i, dtype=torch.long,
                                      device=device)
                    batch.batch.append(item)
                    batch.ptr.append(batch.ptr[-1] + num_nodes)

            batch.batch = None if len(batch.batch) == 0 else batch.batch
            batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
            batch.__slices__ = slices
            batch.__cumsum__ = cumsum
            batch.__cat_dims__ = cat_dims
            batch.__num_nodes_list__ = num_nodes_list

            ref_data = data_list[0]
            for key in batch.keys:
                items = batch[key]
                item = items[0]
                
                ### <--- Updating Cat Dim
                if key in update_cat_dims.keys():
                    cat_dim = update_cat_dims[key]
                else: 
                    cat_dim = ref_data.__cat_dim__(key, item)
                    cat_dim = 0 if cat_dim is None else cat_dim
                ### ---?
                if isinstance(item, Tensor):
                    batch[key] = torch.cat(items, cat_dim)
                elif isinstance(item, SparseTensor):
                    batch[key] = cat(items, cat_dim)
                elif isinstance(item, (int, float)):
                    batch[key] = torch.tensor(items)

            if torch_geometric.is_debug_enabled():
                batch.debug()

            return batch.contiguous()


#---->
class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[:,2:]


class TcgagraphData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data,label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f'fold{self.fold}.csv'
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        #---->order
        self.shuffle = self.dataset_cfg.data_shuffle
        self.transform = ToTensor()

        #---->
        if state == 'train':
            self.data = self.slide_data['train_slide_id'].dropna()
            self.survival_months = self.slide_data['train_survival_months'].dropna()
            self.censorship = self.slide_data['train_censorship'].dropna()
            self.case_id = self.slide_data['train_case_id'].dropna()
            self.label = self.slide_data['train_disc_label'].dropna()
        if state == 'val':
            self.data = self.slide_data['val_slide_id'].dropna()
            self.survival_months = self.slide_data['val_survival_months'].dropna()
            self.censorship = self.slide_data['val_censorship'].dropna()
            self.case_id = self.slide_data['val_case_id'].dropna()
            self.label = self.slide_data['val_disc_label'].dropna()
        if state == 'test':
            self.data = self.slide_data['test_slide_id'].dropna()
            self.survival_months = self.slide_data['test_survival_months'].dropna()
            self.censorship = self.slide_data['test_censorship'].dropna()
            self.case_id = self.slide_data['test_case_id'].dropna()
            self.label = self.slide_data['test_disc_label'].dropna()

        #---->
        splits = [self.data, self.survival_months, self.censorship, self.case_id, self.label]
        self.split_data = pd.concat(splits, ignore_index = True, axis=1)
        self.split_data.columns = ['slide_id', 'survival_months', 'censorship', 'case_id', 'disc_label']

        #---->
        self.patient_df = self.split_data.drop_duplicates(['case_id']).copy()
        self.patient_df.set_index(keys='case_id', drop=True, inplace=True)
        self.split_data.set_index(keys='case_id', drop=True, inplace=True)

        #---->
        self.patient_dict = {}
        for patient in self.patient_df.index:
            slide_ids = self.split_data.loc[patient, 'slide_id'] #
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            self.patient_dict.update({patient:slide_ids}) #
        

    def __len__(self):
        return len(self.patient_df)

    def __getitem__(self, idx):
        case_id = self.case_id[idx]
        event_time = self.survival_months[idx]
        censorship = self.censorship[idx]
        label = self.label[idx]
        slide_ids = self.patient_dict[case_id].tolist()

        features = []
        for slide_id in slide_ids:
            full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            try:
                features.append(torch.load(full_path))
            except:
                print(full_path)
        features = BatchWSI.from_data_list(features, update_cat_dims={'edge_latent': 1})
        # features = torch.cat(features, dim=0)


        #----> 
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label, event_time, censorship


#---->
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

if __name__ == '__main__':
    cfg = read_yaml('GBMLGG/PatchGCN.yaml')
    Mydata = TcgagraphData(dataset_cfg=cfg.Data, state='train')
    dataloader = data.dataloader(Mydata)
    for i, data in enumerate(dataloader):
        pass
