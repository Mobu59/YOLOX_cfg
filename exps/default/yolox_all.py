#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp
from config import *
import sys

#get model configs
#cfg = get_cfg("goods_det")
cfg = get_cfg(sys.argv[2])

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = cfg['num_classes']
        self.depth = cfg['depth']
        self.width = cfg['width']
        self.warmup_epochs = cfg['warmup_epochs']
        self.no_aug_epochs = cfg['no_aug_epochs']
        self.max_epoch = cfg['max_epoch']
        
        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = cfg['flip_prob']
        self.degrees = cfg["degrees"]
        self.test_conf = cfg["test_conf"]
        self.nmsthre = cfg["nmsthre"]
        self.input_size = cfg['input_size']
        self.test_size = cfg['test_size']
        self.mosaic_scale = cfg['mosaic_scale']
        self.enable_mixup = cfg['enable_mixup']
        self.basic_lr_per_img = cfg['lr']
        self.eval_interval = cfg['eval_interval']
        self.output_dir = cfg['output_dir']

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            GoodsDataset,
            TPDataset,    
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = TPDataset(
                    data_dir=cfg['train_data_dir'],
                    img_size=self.input_size,
                    preproc=TrainTransform(
                    max_labels=cfg["max_labels"],
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                    )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import VOCDetection, ValTransform, TPDataset

        valdataset = TPDataset(
                data_dir=cfg['val_data_dir'],
                img_size=self.test_size,
                preproc=ValTransform(legacy=legacy),
                )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
