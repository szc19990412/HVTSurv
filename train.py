# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
import argparse
from pathlib import Path
import numpy as np
import glob

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->设置参数
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='',type=str) 
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--fold', default = 2)
    parser.add_argument('--mask', action='store_true', default=False)
    parser.add_argument('--mask-ratio', default = 0.0, type=float)
    args = parser.parse_args()
    return args

#---->main function
def main(cfg):

    #---->Initialize the seed
    pl.seed_everything(cfg.General.seed)

    #---->加载loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->加载callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->定义Data类
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)

    #---->定义Model类
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->实例化Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, #直接训练
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        gpus=cfg.General.gpus,
        amp_level=cfg.General.amp_level,  # 优化等级
        precision=cfg.General.precision,  # 半精度训练
        # accelerator=cfg.General.multi_gpu_mode,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
        # plugins="ddp_sharded",
        # # resume_from_checkpoint = , #加载预训练模型
        # limit_train_batches=0.02, #调试代码用
        # limit_val_batches=1, #调试代码用
    )

    #---->训练或者测试
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    #---->配置参数
    args = make_parse()
 
    #---->读取yaml配置
    cfg = read_yaml(args.config)

    #---->update: 将args的参数保存到cfg中
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    if args.mask == True:
        cfg.Model.mask_ratio = args.mask_ratio

    #---->main函数
    main(cfg)
 