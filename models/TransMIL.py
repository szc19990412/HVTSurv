# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from numpy.core.fromnumeric import squeeze
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from nystrom_attention import Nystromformer
from timm.models.layers import trunc_normal_
# x,residual  [B,C,H,W]


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=128):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PEG(nn.Module):
    def __init__(self, dim=256, k=7):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)


    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


# It's challenging for TransMIL to process all the high-dimensional data in the patient-level bag, so we reduce the dimension from 1024 to 128.
class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PEG(128)
        self._fc1 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=128)
        self.layer2 = TransLayer(dim=128)
        self.norm = nn.LayerNorm(128)
        self._fc2 = nn.Linear(128, self.n_classes)


    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        
        #---->Dimensionality reduction first
        h = self._fc1(h) #[B, n, 128]
        
        #---->padding
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 128]

        #---->Add position encoding, after a transformer
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 128]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 128]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 128]

        h = self.norm(h)[:,0]

        #---->predict output
        logits = self._fc2(h)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
        return results_dict

