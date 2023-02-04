
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.vision_transformer import Mlp
from einops import rearrange, reduce
from torch import nn, einsum
from timm.models.layers import trunc_normal_
import math

class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


# Thanks to open-source implementation of iRPE: https://github.com/microsoft/Cream/blob/76d03f6f7438388855df1cad62741721f990f9ac/iRPE/DETR-with-iRPE/models/rpe_attention/irpe.py#L19
@torch.no_grad()
def piecewise_index(relative_position, alpha=1.9, beta=1.9*4, gamma=1.9*6, shift=7, dtype=torch.int32):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha*2
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - 2*alpha)).round().clip(max=shift)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    idx[mask] = torch.sign(idx[mask])*1
    return idx

#---->WindowAttention
# Thanks to open-source implementation of Swin-Transformer: https://github.com/microsoft/Swin-Transformer/blob/d19503d7fbed704792a5e5a3a5ee36f9357d26c1/models/swin_transformer.py#L77 
class WindowAttention(nn.Module):
    
    def __init__(self, dim=512, window_size=49, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = int(np.sqrt(window_size))
        self.num_heads = 8
        head_dim = dim // 8
        self.scale = head_dim ** -0.5

        #---->RelativePosition
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*self.shift+1, self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        #---->Attention
        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, coords):
        B, N, C = x.shape #[b, n, c]

        #---->partition windows
        x = rearrange(x, 'b (w ws) c -> b w ws c', ws=self.window_size)
        x = rearrange(x, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]

        coords = rearrange(coords, 'b (w ws) c -> b w ws c', ws=self.window_size)
        coords = rearrange(coords, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]
        

        #---->attention
        B_, N, C = x.shape
        #[b*num_window, window, 3*C]->[b*num_window, window, 3, num_head, C//num_head]->[3, b*num_window, num_head, window, C//num_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2] #[b*num_window, num_head, window, C//num_head]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[num_window, num_head, window, window]


        max_L = coords.shape[1] #[num_window, window_size, 2]
        relative_coords = coords.view((-1, max_L, 1, 2))-coords.view((-1, 1, max_L, 2))
        relative_coords = relative_coords.int()
        relative_coords[:, :, :, 0] = piecewise_index(relative_coords[:, :, :, 0], shift=self.shift)
        relative_coords[:, :, :, 1] = piecewise_index(relative_coords[:, :, :, 1], shift=self.shift)
        relative_coords = relative_coords.abs()

        relative_position_index = relative_coords.sum(-1)  # num_window, Wh*Ww, Wh*Ww
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(-1, self.window_size, self.window_size, self.num_heads)
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()

        attn = attn + relative_position_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = (attn @ v)
        
        x = out.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        #---->window_reverse
        x = rearrange(x, '(b w) ws c -> b (w ws) c', b=B)
        return x

#---->WindowAttention
class ShuffleWindowAttention(nn.Module):
    
    def __init__(self, dim=512, window_size=49, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = 8
        head_dim = dim // 8
        self.scale = head_dim ** -0.5

        #---->Attention
        self.qkv = nn.Linear(dim, dim*3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        B, N, C = x.shape #[b, n, c]

        #---->partition windows
        # Thanks to open-source implementation of Shuffle-Transformer: https://github.com/mulinmeng/Shuffle-Transformer/blob/8ba81eacf01314d4d26ff514f61badc7eebd33de/models/shuffle_transformer.py#L69
        x = rearrange(x, 'b (ws w) c -> b w ws c', ws=self.window_size)
        x = rearrange(x, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]
        #---->attention
        B_, N, C = x.shape

        #[b*num_window, window, 3*C]->[b*num_window, window, 3, num_head, C//num_head]->[3, b*num_window, num_head, window, C//num_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #[b*num_window, num_head, window, C//num_head]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[num_window, num_head, window, window]

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, '(b w) ws c -> b (ws w) c', b=B)
        return x


class LocalLayer(nn.Module):
    
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, window_size=49):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.wattn = WindowAttention(dim=dim, window_size=window_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x, coords):
        #---->pad
        h_ = x.shape[1]
        add_length = (h_//self.window_size)*self.window_size - h_ 
        if add_length != 0:
            x = rearrange(x, 'b n c -> b c n')
            coords = rearrange(coords, 'b n c -> b c n')
            #---->feature
            x = F.pad(input=x, pad=(add_length//2, add_length-add_length//2), mode='reflect') #镜像
            x = rearrange(x, 'b c n -> b n c')
            #---->coords
            coords = F.pad(input=coords, pad=(add_length//2, add_length-add_length//2), mode='reflect') #镜像
            coords = rearrange(coords, 'b c n -> b n c')
        #---->windowd
        x = x + self.wattn(self.norm1(x), coords)
        x = self.act(x)
        x = self.drop(x)
        return x

class HVTSurv(nn.Module):
    def __init__(self, dropout = False, n_classes=4, window_size=49):
        super(HVTSurv, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.n_classes = n_classes
        self.layer1 = LocalLayer(dim=512, window_size=window_size)
        self._fc2 = nn.Linear(512, self.n_classes)
        # shuffle
        self.shiftwattn = ShuffleWindowAttention(dim=512, window_size=window_size)
        self.mlp1 = Mlp(in_features=512, hidden_features=512, act_layer=nn.GELU, drop=0.1)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(512)
        self.attnpool = Attn_Net(L = 512, D = 256, dropout = dropout, n_classes = 1)
        

    def forward(self, **kwargs):
    
        h_all = kwargs['data'] #list [[B,n,1024],...,[],[]]

        feature_patient = []
        for h in h_all: #All WSIs corresponding to a patient

            #---->Separate feature, coords information
            coords = h[:, :, :2].clone() #[n, 2]
            h = h[:, :, 2:] #[n, 1024]

            #---->Dimensionality reduction
            h = self._fc1(h) #[B, n, 512]

            #---->LocalLayer
            feature = self.layer1(h, coords)
            
            #---->shufflewindow
            feature = feature + self.shiftwattn(self.norm1(feature))
            feature = feature + self.mlp1(self.norm2(feature))

            feature_patient.append(feature)
        #---->concat
        feature = torch.cat(feature_patient, dim=1)
        #---->patient-level attention
        feature = self.norm3(feature) # B, N, C
        A, feature = self.attnpool(feature.squeeze(0))  # B C 1
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        feature = torch.mm(A, feature)

        #---->predict output
        logits = self._fc2(feature)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
        return results_dict
