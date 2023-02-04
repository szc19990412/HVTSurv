### System
import os, sys
from os.path import join
import h5py
import math
from math import floor
import torch
from time import time
from tqdm import tqdm
import argparse

### Numerical Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

### Graph Network Packages
import nmslib



#Thanks to open-source implementation of WSI-Graph Construction in PatchGCN: https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=False):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


def pt2graph(wsi_h5, radius=49):
    coords, features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert coords.shape[0] == features.shape[0]

    #---->padding
    window_size = radius

    h_ = features.shape[0]
    add_length = (h_//window_size+1)*window_size - h_
    #---->feature
    features = np.pad(array=features, pad_width=((add_length//2, add_length-add_length//2),(0,0)), mode='reflect') #reflect padding
    #---->coords
    coords = np.pad(array=coords, pad_width=((add_length//2, add_length-add_length//2),(0,0)), mode='reflect') #reflect padding
    #---->The value of the coordinate divided by the size of the patch
    coords = coords//256

    #---->Get the minimum value of the coordinates
    coords_min = np.min(coords, axis=0)
    coords = coords - coords_min + 1 #The horizontal and vertical coordinates start counting from 1


    num_patches = coords.shape[0]
    
    #---->Recombine feature order
    features_knn = []
    coords_knn = []
    max_distance = []
    for v_idx in range(0, num_patches, window_size):
        model = Hnsw(space='l2')
        model.fit(coords)
        select_index = model.query(coords[0], topn=radius) #Choose 48 adjacent ones including yourself, a total of 49
        features_knn.extend(features[select_index])
        coords_knn.extend(coords[select_index])

        max_distance.append(np.max((coords[select_index][np.newaxis, :, :]-coords[select_index][:, np.newaxis, :]).sum(-1)))

        #---->delete selected features
        features = np.delete(features, select_index, axis=0)
        coords = np.delete(coords, select_index, axis=0)


    features = torch.from_numpy(np.stack(features_knn, axis=0)).to(torch.float32)
    coords = torch.from_numpy(np.stack(coords_knn, axis=0))
    #---->concat features and coordinates
    features = torch.cat((coords, features), dim=1)
    return features

def createDir_h5toPyG(h5_path, save_path, radius):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))

        try:
            if os.path.exists(os.path.join(save_path, h5_fname[:-3]+'.pt')):
                continue
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = pt2graph(wsi_h5, radius)
            torch.save(G, os.path.join(save_path, h5_fname[:-3]+'.pt'))
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-path', type=str, default= '/h5_files/')
    parser.add_argument('--save-path', type=str, default= '/pt_knn/')
    parser.add_argument('--radius', type=int, default= 49)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_parse()
    h5_path = args.h5_path
    save_path = args.save_path
    radius = args.radius
    os.makedirs(save_path, exist_ok=True)
    createDir_h5toPyG(h5_path, save_path, radius)
