# HVTSurv
HVTSurv: Hierarchical Vision Transformer for Patient-level Survival Prediction from Whole Slide Image-AAAI 2023


## Installation

- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 3090)
- Python (3.7.11), pytorch-lightning (1.2.3), PyTorch (1.7.1), torchvision (0.8.2), timm (0.3.2)

Please refer to the following instructions.

```python
# create and activate the conda environment
conda create -n hvtsurv python=3.7
conda activate hvtsurv

# install pytorch
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/cu110/torch_stable.html

# install related package
pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-sparse==0.6.9 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==2.0.1

pip install -r requirements.txt
```


## Feature Generation

### WSI processing and label processing

Please refer to the [CLAM](https://github.com/mahmoodlab/CLAM) to embed WSIs into features, and refer to [PatchGCN](https://github.com/mahmoodlab/Patch-GCN) and [PORPOISE](https://github.com/mahmoodlab/PORPOISE) to process the label of follow-up time and censorship.

### Feature rearrangement

To better reflect the local characteristics in both horizontal and vertical directions within a window, a feature rearrangement method is proposed to ensure the closeness in both directions of the 2D space after the window partition.

```python
python knn_position.py --h5-path='TCGA_xxxx/h5_files/' --save-path='TCGA_xxxx/pt_knn/'
```

### Random window masking

To increase the robustness of the model for tumor heterogeneity and further exploit the advantages of our hierarchical processing framework, we propose a random window masking strategy. A WSI bag will be further split into several sub-WSI bags.

```python
python random_mask_window.py --pt_dir='TCGA_xxxx/pt_knn/' --csv_dir='splits/4foldcv/tcga_xxxx/' --window_size=49 --num_bag=2 --masking_ratio=0.5 --seed=42
```

The folder structure is as follows:

```python
TCGA_xxxx/
    └──h5_files/
        ├── slide_1.h5
        ├── slide_2.h5
        └── ...
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──pt_graph/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──pt_graph/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──pt_knn/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──pt_knn_2_0.5_random/
        ├── slide_1_0.pt
      	├── slide_1_1.pt
        ├── slide_2_0.pt
        ├── slide_2_1.pt
        └── ...
```



### Feature Aggregation

```python
for((FOLD=0;FOLD<4;FOLD++));
do
    python train.py --stage='train'\
    --config='BRCA/HVTSurv.yaml'  --gpus=0 --fold=$FOLD
    python train.py --stage='test'\
    --config='BRCA/HVTSurv.yaml'  --gpus=0 --fold=$FOLD
done
python metrics.py --config='BRCA/HVTSurv.yaml'
python curves.py --config='BRCA/HVTSurv.yaml'
```

