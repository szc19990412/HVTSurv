#further split a patient bag into many small bags

import torch 
import argparse
import glob
from pathlib import Path
from einops import rearrange, reduce
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--pt_dir', type=str, default='/data112/shaozc/TCGA_feature/BLCA/FEATURES_DIRECTORY/tcga_blca_resnet_features/pt_knn/' )
parser.add_argument('--csv_dir', type=str, default='/data112/shaozc/HVTSurv/splits/4foldcv/tcga_blca/' )
parser.add_argument('--window_size', type=int, default=49)
parser.add_argument('--num_bag', type=int, default=2)
parser.add_argument('--masking_ratio', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()


def random_masking(x, coords):
    N = len(coords)
    S = int(N * (1 - args.masking_ratio))
    index = torch.LongTensor(np.random.choice(
        range(N), S, replace=True)).to(x.device) #Allows oversampling of data
    x_out = torch.index_select(x, 0, index)
    coords = torch.index_select(coords, 0, index)
    return x_out, coords


if __name__ == '__main__':
    #---->
    pt_dir_all = glob.glob(args.pt_dir + '*.pt')

    #---->
    save_dir = Path(args.pt_dir).parent / f'{Path(args.pt_dir).stem}_{args.num_bag}_{args.masking_ratio}_random'
    Path(save_dir).mkdir(exist_ok=True, parents=True)

    np.random.seed(args.seed)

    #---->
    csv_save_dir = Path(args.csv_dir).parent / f'{Path(args.csv_dir).stem}_{args.num_bag}'
    csv_save_dir.mkdir(exist_ok=True, parents=True)
    csv_dir_all = glob.glob(args.csv_dir + '*.csv')
    for csv_dir in csv_dir_all:
        train_slide_id, train_survival_months, train_censorship, train_case_id, train_disc_label = [],[],[],[],[]
        val_slide_id, val_survival_months, val_censorship, val_case_id, val_disc_label = [],[],[],[],[]
        test_slide_id, test_survival_months, test_censorship, test_case_id, test_disc_label = [],[],[],[],[]
        df = pd.read_csv(csv_dir, index_col=0)
        df_train = df.iloc[:, :5].dropna(axis=0)
        df_train.index = df_train['train_slide_id']
        df_val = df.iloc[:, 5:10].dropna(axis=0)
        df_val.index = df_val['val_slide_id']
        df_test = df.iloc[:, 10:15].dropna(axis=0)
        df_test.index = df_test['test_slide_id']
        #---->Add bag according to slide_id
        train_id = df_train['train_slide_id'].tolist()
        val_id = df_val['val_slide_id'].tolist()
        test_id = df_test['test_slide_id'].tolist()

        for id in train_id:
            for i in range(args.num_bag):
                train_slide_id.append(f'{id}_{i}')
                train_survival_months.append(df_train.loc[id, 'train_survival_months'])
                train_censorship.append(df_train.loc[id, 'train_censorship'])
                train_case_id.append(df_train.loc[id, 'train_case_id'])
                train_disc_label.append(df_train.loc[id, 'train_disc_label'])
        for id in val_id:
            for i in range(args.num_bag):
                val_slide_id.append(f'{id}_{i}')
                val_survival_months.append(df_val.loc[id, 'val_survival_months'])
                val_censorship.append(df_val.loc[id, 'val_censorship'])
                val_case_id.append(df_val.loc[id, 'val_case_id'])
                val_disc_label.append(df_val.loc[id, 'val_disc_label'])
        for id in test_id:
            for i in range(args.num_bag):
                test_slide_id.append(f'{id}_{i}')
                test_survival_months.append(df_test.loc[id, 'test_survival_months'])
                test_censorship.append(df_test.loc[id, 'test_censorship'])
                test_case_id.append(df_test.loc[id, 'test_case_id'])
                test_disc_label.append(df_test.loc[id, 'test_disc_label'])

        #---->Get the slide_id of train, val.test
        train_slide_id, train_survival_months, train_censorship, train_case_id, train_disc_label = \
            pd.Series(train_slide_id), pd.Series(train_survival_months), pd.Series(train_censorship), pd.Series(train_case_id), pd.Series(train_disc_label)
        val_slide_id, val_survival_months, val_censorship, val_case_id, val_disc_label = \
            pd.Series(val_slide_id), pd.Series(val_survival_months), pd.Series(val_censorship), pd.Series(val_case_id), pd.Series(val_disc_label)
        test_slide_id, test_survival_months, test_censorship, test_case_id, test_disc_label = \
            pd.Series(test_slide_id), pd.Series(test_survival_months), pd.Series(test_censorship), pd.Series(test_case_id), pd.Series(test_disc_label)

        splits = [train_slide_id, train_survival_months, train_censorship, train_case_id, train_disc_label,\
                  val_slide_id, val_survival_months, val_censorship, val_case_id, val_disc_label,\
                  test_slide_id, test_survival_months, test_censorship, test_case_id, test_disc_label
            ]
        #---->Save this fold information
        df = pd.concat(splits, ignore_index = True, axis=1)
        df.columns = ['train_slide_id', 'train_survival_months', 'train_censorship', 'train_case_id', 'train_disc_label',\
                  'val_slide_id', 'val_survival_months', 'val_censorship', 'val_case_id', 'val_disc_label',\
                  'test_slide_id', 'test_survival_months', 'test_censorship', 'test_case_id', 'test_disc_label'
            ]
        csv_path = csv_save_dir / f'{Path(csv_dir).name}'
        df.to_csv(csv_path)


    for pt_dir in tqdm(pt_dir_all):

        #---->
        num_pt_dir = glob.glob(f'{Path(save_dir)/Path(pt_dir).stem}*.pt')
        if len(num_pt_dir) == args.num_bag:
            continue

        feature_all = torch.load(pt_dir)

        #---->
        coords = feature_all[:, :2].clone() #[n, 2]
        x = feature_all[:, 2:] #[n, 1024]

        #---->pad
        h_ = x.shape[0]
        add_length = (h_//args.window_size+1)*args.window_size - h_ 
        if add_length != 0:
            x = rearrange(x, 'n c -> c n')
            coords = rearrange(coords, 'n c -> c n')
            #---->feature
            x = F.pad(input=x.unsqueeze(0), pad=(add_length//2, add_length-add_length//2), mode='reflect').squeeze(0) 
            x = rearrange(x, 'c n -> n c')
            #---->coords
            coords = F.pad(input=coords.unsqueeze(0), pad=(add_length//2, add_length-add_length//2), mode='reflect').squeeze(0) 
            coords = rearrange(coords, 'c n -> n c')


        #---->partition windows
        x = rearrange(x, '(w ws) c -> w ws c', ws=args.window_size)

        coords = rearrange(coords, '(w ws) c -> w ws c', ws=args.window_size)

        for i in range(args.num_bag):
            #---->random mask
            x_out, coords_out = random_masking(x, coords)
            #---->restore window
            x_out = rearrange(x_out, 'w ws c -> (w ws) c')
            coords_out = rearrange(coords_out, 'w ws c -> (w ws) c')
            #----concat
            feature_bag = torch.cat((coords_out, x_out), dim=-1)
            torch.save(feature_bag, Path(save_dir)/ f'{Path(pt_dir).stem}_{i}.pt')
