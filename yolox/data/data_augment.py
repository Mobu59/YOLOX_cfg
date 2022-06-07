#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random
import sys

import cv2
import torch
import numpy as np

from yolox.utils import xyxy2cxcywh
from config import *

cfg = get_cfg(sys.argv[2])

def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
        #return value
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
                          or single float values. Got {}".format(
                value
            )
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    #R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
    #R = cv2.getRotationMatrix2D(angle=angle, center=(int(twidth/2), int(theight/2)), scale=scale)
    #M = R

    M = np.ones([2, 3])
    angle = angle * np.pi / 180
    tx = (1 - np.cos(angle)) * (twidth/2) - np.sin(angle)*(theight/2)
    ty = (1 - np.cos(angle)) * (theight/2) + np.sin(angle)*(twidth/2)
    M[0,0] = np.cos(angle)
    M[0,1] = np.sin(angle)
    M[1,0] = -np.sin(angle)
    M[1,1] = np.cos(angle)
    M[0,2] = tx
    M[1,2] = ty
    bound_w = int(twidth * np.abs(np.cos(angle)) + theight * np.abs(np.sin(angle)))
    bound_h = int(twidth * np.abs(np.sin(angle)) + theight * np.abs(np.cos(angle)))
    M[0,2] += bound_w/2 -(twidth/2)
    M[1,2] += bound_h/2 -(theight/2)
    ## Shear
    #shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    #shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    #M[0] = R[0] + shear_y * R[1]
    #M[1] = R[1] + shear_x * R[0]

    ## Translation
    #translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    #translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    #M[0, 2] = translation_x
    #M[1, 2] = translation_y

    return M, scale, angle, bound_w, bound_h


def apply_affine_to_bboxes(targets, target_size, M, scale, angle):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )
    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)
    
    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale, angle, bw, bh = get_affine_matrix(target_size, degrees, translate, scales, shear)

    #img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))
    img = cv2.warpAffine(img, M, dsize=(bw, bh), borderValue=(114, 114, 114))
    padded_img = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 114
    r = min(target_size[1] / img.shape[0], target_size[0] / img.shape[1])
    try:
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    except cv2.error as e:
        print(e)
        pass
    img = padded_img
    # Transform label coordinates
    if len(targets) > 0:
        #targets = apply_affine_to_bboxes(targets, target_size, M, scale, angle)
        targets = apply_affine_to_bboxes(targets, (bw, bh), M, scale, angle)
        targets = targets * r

    return img, targets


def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def brightness_aug(image):
    image_copy = image.copy()
    image_copy = image_copy.astype(np.float32) / 255.0
    lightness = random.uniform(-20, 20)
    contrass = random.uniform(20, 40)
    hls_img = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HLS)
    hls_img[:, :, 1] = (1.0 + lightness / 100.0) * hls_img[:, :, 1]
    hls_img[:, :, 1][hls_img[:, :, 1] > 1] =1

    hls_img[:, :, 2] = (1.0 + contrass / 100.0) * hls_img[:, :, 2]
    hls_img[:, :, 2][hls_img[:, :, 2] > 1] =1

    new_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2BGR) * 255
    new_img = new_img.astype(np.uint8)

    return new_img       


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    #r = min(208 / img.shape[0], 208 / img.shape[1])
    try:
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    except cv2.error as e:
        print(e)
        pass
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        #image = brightness_aug(image)    
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        #mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 8
        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 2
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        #-----------filter boxes with incorrect w and h-------------
        save_boxes = []
        index = []
        for i, k in enumerate(boxes_t):
            if k[2]/k[3] > cfg["aspect_ratio"] or k[3]/k[2] > cfg["aspect_ratio"]:
                continue
            index.append(i)
            save_boxes.append(k)
        boxes_t = np.ascontiguousarray(save_boxes)
        labels_t = labels_t[index]
        #------------------------------------------------------

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            #boxes_o *= r_o
            boxes_o *= 0
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))
