import inspect # 
import importlib # In order to dynamically import the library
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
#gcn
from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch

#---->
def collate_MIL_survival_graph(batch):
    elem = batch[0]
    elem_type = type(elem)        
    transposed = zip(*batch)
    return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for samples in transposed]


#########DTMIL
from typing import Optional, List
from torch import Tensor
import torch
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def collate_fn(batch):
    batch = list(zip(*batch))

    # in wsi, it is patch bag N * C * H * W, mask 1 * N
    batch[0] = torch.stack(batch[0][0])
    batch[1] = torch.stack(batch[1][0])

    tens = NestedTensor(batch[0], batch[1])
    return tens, default_collate([batch[2], batch[3], batch[4]])
##########

class DataInterface(pl.LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.load_data_module()

 

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.instancialize(state='train')
            self.val_dataset = self.instancialize(state='val')
 

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = self.instancialize(state='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True, collate_fn=collate_MIL_survival_graph)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False, collate_fn=collate_MIL_survival_graph)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False, collate_fn=collate_MIL_survival_graph)


    def load_data_module(self):
        """  
        py文件命名为xx_data, 导入xx_data的XxData, 保存在self.data_module中
        """
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)