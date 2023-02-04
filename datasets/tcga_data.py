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
#---->
class RemoveCoordinates(object):
    """Remove tile levels and coordinates."""
    def __call__(self, sample):
        return sample[:,2:]

class ToTensor(object):
    """A simple transformation on numpy array to obtain torch-friendly tensors.
    """
    def __init__(self, n_tiles=300000):
        self.n_tiles = n_tiles

    def __call__(self, sample):
        # x = torch.from_numpy(sample).float()
        x = sample.float()
        if x.shape[0] > self.n_tiles: #tiles数不到8000，后续补0
            index = [i for i in range(x.shape[0])]
            index = random.sample(index, self.n_tiles)
            x = x[index, :]
        # elif x.shape[0] < self.n_tiles:
        #     x = torch.cat((x, torch.zeros((self.n_tiles - x.shape[0], 2050))))
        return x

class TcgaData(data.Dataset):
    def __init__(self, dataset_cfg=None,
                 state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        #---->data和label
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

        #---->Concat related information together
        splits = [self.data, self.survival_months, self.censorship, self.case_id, self.label]
        self.split_data = pd.concat(splits, ignore_index = True, axis=1)
        self.split_data.columns = ['slide_id', 'survival_months', 'censorship', 'case_id', 'disc_label']

        #---->get patient data
        self.patient_df = self.split_data.drop_duplicates(['case_id']).copy()
        self.patient_df.set_index(keys='case_id', drop=True, inplace=True)
        self.split_data.set_index(keys='case_id', drop=True, inplace=True)

        #---->Establish a connection between patient_df and data
        self.patient_dict = {}
        for patient in self.patient_df.index:
            slide_ids = self.split_data.loc[patient, 'slide_id'] #取出case_id
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            self.patient_dict.update({patient:slide_ids}) #更新字典，每个patient包括哪些Slide
        

    def __len__(self):
        return len(self.patient_df)

    def __getitem__(self, idx):
        case_id = self.case_id[idx]
        event_time = self.survival_months[idx]
        censorship = self.censorship[idx]
        label = self.label[idx]
        slide_ids = self.patient_dict[case_id]

        #---->Load all WSI data under a patient
        features = []
        for slide_id in slide_ids:
            full_path = Path(self.feature_dir) / f'{slide_id}.pt'
            features.append(torch.load(full_path))
        features = torch.cat(features, dim=0)
        features = self.transform(features)

        #----> Shuffle the input order
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
    cfg = read_yaml('BRCA/HVTSurv.yaml')
    Mydata = TcgaData(dataset_cfg=cfg.Data, state='train')
    dataloader = data.dataloader(Mydata)
    for i, data in enumerate(dataloader):
        pass
