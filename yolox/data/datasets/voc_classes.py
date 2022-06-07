#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from config import *
import sys

#cfg = get_cfg("goods_det")
cfg = get_cfg(sys.argv[2])
# VOC_CLASSES = ( '__background__', # always index 0
#VOC_CLASSES = (
#    "0",
#    #"1",
#    #"2",
#    #"3",
#    #"4",
#    #"5",
#    #"6",
#    #"7",
#    #"8",
#)
VOC_CLASSES = tuple()
VOC_CLASSES += cfg['classes']

