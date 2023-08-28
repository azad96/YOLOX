#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/2022-10-06T16-34-42"
        
        # self.train_ann = "coco_annotations.json"
        # self.val_ann = "coco_annotations.json"

        # self.train_ann = "coco_annotations_only_car.json"
        # self.val_ann = "coco_annotations_only_car.json"

        # self.train_ann = "subsampled_coco_annotations.json"
        # self.val_ann = "subsampled_coco_annotations.json"

        self.train_ann = "coco_annotations_only_car_no_empty_train.json"
        self.val_ann = "coco_annotations_only_car_no_empty_val.json"

        self.input_size = (640, 640)
        self.num_classes = 1

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        self.print_interval = 50

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            name="images",
            cache=cache,
            cache_type=cache_type,
        )