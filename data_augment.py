from __future__ import print_function, division

import cv2
import numpy as np
import random
import albumentations as albu

from PIL import Image
from facedet.utils.box_utils import matrix_iof

def get_bi_headtopboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_bi_headtopboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=50, quality_upper=80,
                always_apply=False, p=0.5),
            # albu.augmentations.transforms.RandomScale(
            #     scale_limit=(0.5, 1.2), p=0.5),
            # albu.PadIfNeeded(min_height=height,
            #                  always_apply=True, border_mode=cv2.BORDER_CONSTANT,
            #                  value=[0, 0, 0]),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            # albu.augmentations.transforms.RandomCrop(
            #     height=height,
            #     width=width, p=1.0),
            albu.augmentations.transforms.RandomResizedCrop(
                scale=(0.5, 1.2), ratio=(0.9, 1.2),
                height=height,
                width=width, p=1.0),
            albu.augmentations.transforms.Rotate(
                limit=180, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
                mask_value=[0, 0, 0], always_apply=False, p=1.0),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.5),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.Blur(blur_limit=7, p=0.5),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=50,
                                        sat_shift_limit=50),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            albu.HorizontalFlip(p=0.5),

            albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['bbox_index']))
    # return albu.Compose(
    #         list_transforms,
    #         bbox_params=albu.BboxParams(
    #             format=bbox_format,
    #             min_area=min_area,
    #             min_visibility=min_visibility,
    #             label_fields=['bbox_index']),
    #         keypoint_params=albu.KeypointParams(
    #             format='xy',
    #             # label_fields=['kp_label'],
    #             remove_invisible=False))
def get_goodboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_goodboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=50, quality_upper=80,
                always_apply=False, p=0.5),
            # albu.augmentations.transforms.RandomScale(
            #     scale_limit=(0.5, 1.2), p=0.5),
            # albu.PadIfNeeded(min_height=height,
            #                  always_apply=True, border_mode=cv2.BORDER_CONSTANT,
            #                  value=[0, 0, 0]),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            albu.augmentations.transforms.RandomCrop(
                height=height,
                width=width, p=1.0),
            # albu.augmentations.transforms.RandomResizedCrop(
            #     height=height,
            #     width=width, p=0.2),
            # albu.augmentations.transforms.Rotate(
            #     limit=15, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
            #     value=[0, 0, 0],
            #     mask_value=[0, 0, 0], always_apply=False, p=0.5),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.2,
                                              contrast_limit=0.2),
                # albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.Blur(blur_limit=7, p=0.5),
            albu.OneOf([
                # albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
                #               g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=10,
                                        sat_shift_limit=10),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            # albu.HorizontalFlip(p=0.15),

            # albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['bbox_index']))
    # return albu.Compose(
    #         list_transforms,
    #         bbox_params=albu.BboxParams(
    #             format=bbox_format,
    #             min_area=min_area,
    #             min_visibility=min_visibility,
    #             label_fields=['bbox_index']),
    #         keypoint_params=albu.KeypointParams(
    #             format='xy',
    #             # label_fields=['kp_label'],
    #             remove_invisible=False))
def get_bi_headboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_bi_headboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=20, quality_upper=80,
                always_apply=False, p=0.5),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            # albu.PadIfNeeded(min_height=int(height*1.2),
            #                  min_width=int(width*1.2),
            #                  always_apply=True, border_mode=cv2.BORDER_CONSTANT,
            #                  value=[0, 0, 0]),
            # albu.augmentations.transforms.RandomScale(
            #     scale_limit=(0.8, 1), p=1.0),
            albu.augmentations.transforms.Rotate(
                limit=15, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
                mask_value=[0, 0, 0], always_apply=False, p=0.5),
            albu.augmentations.transforms.RandomCrop(
                height=height,
                width=width, p=1.0),
            # albu.augmentations.transforms.RandomResizedCrop(
            #     height=height,
            #     width=width, p=0.2),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.3,
                                              contrast_limit=0.3),
                albu.RandomGamma(gamma_limit=(10, 50)),
                albu.NoOp()
            ]),
            albu.augmentations.transforms.Blur(blur_limit=7, p=0.5),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=10,
                                        sat_shift_limit=10),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            # albu.HorizontalFlip(p=0.15),

            # albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['bbox_index']))
    # return albu.Compose(
    #         list_transforms,
    #         bbox_params=albu.BboxParams(
    #             format=bbox_format,
    #             min_area=min_area,
    #             min_visibility=min_visibility,
    #             label_fields=['bbox_index']),
    #         keypoint_params=albu.KeypointParams(
    #             format='xy',
    #             # label_fields=['kp_label'],
    #             remove_invisible=False))

def get_bi_pedboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_bi_pedboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=50, quality_upper=80,
                always_apply=False, p=0.5),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            # albu.PadIfNeeded(min_height=int(height*1.2),
            #                  min_width=int(width*1.2),
            #                  always_apply=True, border_mode=cv2.BORDER_CONSTANT,
            #                  value=[0, 0, 0]),
            # albu.augmentations.transforms.RandomScale(
            #     scale_limit=(0.8, 1), p=1.0),
            # albu.augmentations.transforms.Rotate(
            #     limit=15, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
            #     value=[0, 0, 0],
            #     mask_value=[0, 0, 0], always_apply=False, p=0.5),
            albu.augmentations.transforms.RandomCrop(
                height=height,
                width=width, p=1.0),
            # albu.augmentations.transforms.RandomResizedCrop(
            #     height=height,
            #     width=width, p=0.2),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            # albu.OneOf([
            #     albu.RandomBrightnessContrast(brightness_limit=0.1,
            #                                   contrast_limit=0.1),
            #     albu.RandomGamma(gamma_limit=(0, 10)),
            #     albu.NoOp()
            # ]),
            # albu.augmentations.transforms.Blur(blur_limit=7, p=0.5),
            # albu.OneOf([
            #     albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
            #                   g_shift_limit=15),
            #     albu.HueSaturationValue(hue_shift_limit=10,
            #                             sat_shift_limit=10),
            #     albu.NoOp()
            # ]),
            # albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            # albu.HorizontalFlip(p=0.15),

            # albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['bbox_index']))
    # return albu.Compose(
    #         list_transforms,
    #         bbox_params=albu.BboxParams(
    #             format=bbox_format,
    #             min_area=min_area,
    #             min_visibility=min_visibility,
    #             label_fields=['bbox_index']),
    #         keypoint_params=albu.KeypointParams(
    #             format='xy',
    #             # label_fields=['kp_label'],
    #             remove_invisible=False))

def get_bi_faceboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_bi_faceboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=30, quality_upper=80,
                always_apply=False, p=0.5),
            albu.augmentations.transforms.RandomScale(
                scale_limit=(0.5, 1.2), p=0.5),
            albu.PadIfNeeded(min_height=height,
                             always_apply=True, border_mode=cv2.BORDER_CONSTANT,
                             value=[0, 0, 0]),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            albu.augmentations.transforms.RandomCrop(
                height=height,
                width=width, p=1.0),
            # albu.augmentations.transforms.RandomResizedCrop(
            #     height=height,
            #     width=width, p=1.0),
            albu.augmentations.transforms.Rotate(
                limit=20, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
                mask_value=[0, 0, 0], always_apply=False, p=0.8),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.5),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.augmentations.transforms.Blur(blur_limit=7, p=0.5),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=50,
                                        sat_shift_limit=50),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            # albu.HorizontalFlip(p=0.15),

            # albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['category_id']))
            # keypoint_params=albu.KeypointParams(
            #     format='xy'))

def get_po_faceboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_po_faceboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=10, quality_upper=30,
                always_apply=False, p=0.5),
            albu.OneOf([
                albu.augmentations.transforms.RandomScale(
                    scale_limit=(0.05, 1.2), p=1.0),
                albu.augmentations.transforms.RandomScale(
                    scale_limit=(0.05, 0.4), p=1.0),
                albu.augmentations.transforms.RandomScale(
                    scale_limit=(0.05, 0.2), p=1.0),
                ]),
            albu.PadIfNeeded(min_height=height,
                             always_apply=True, border_mode=cv2.BORDER_CONSTANT,
                             value=[0, 0, 0]),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            albu.augmentations.transforms.RandomCrop(
                height=height,
                width=width, p=1.0),
            # albu.augmentations.transforms.RandomResizedCrop(
            #     height=height,
            #     width=width, p=1.0),
            albu.augmentations.transforms.Rotate(
                limit=100, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
                mask_value=[0, 0, 0], always_apply=False, p=0.7),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.5),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.augmentations.transforms.Blur(blur_limit=14, p=0.5),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=50,
                                        sat_shift_limit=50),
                albu.NoOp()
            ]),
            albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            # albu.HorizontalFlip(p=0.15),

            # albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['category_id']))
            # keypoint_params=albu.KeypointParams(
            #     format='xy'))

def get_cocheck_faceboxes_augumentation(phase, width=540, height=540, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_cocheck_faceboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=50, quality_upper=80,
                always_apply=False, p=0.3),
            albu.augmentations.transforms.RandomScale(
                scale_limit=(0.5, 1.2), p=0.5),
            albu.PadIfNeeded(min_height=height,
                             always_apply=True, border_mode=cv2.BORDER_CONSTANT,
                             value=[0, 0, 0]),
            albu.augmentations.transforms.SmallestMaxSize(
                max_size=width, always_apply=True),
            albu.augmentations.transforms.RandomCrop(
                height=height,
                width=width, p=1.0),
            # albu.augmentations.transforms.RandomResizedCrop(
            #     height=height,
            #     width=width, p=1.0),
            albu.augmentations.transforms.Rotate(
                limit=100, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
                mask_value=[0, 0, 0], always_apply=False, p=0.5),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5,
                                              contrast_limit=0.5),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=15, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=50,
                                        sat_shift_limit=50),
                albu.NoOp()
            ]),
            albu.augmentations.transforms.ToGray(always_apply=False, p=0.1),
            # albu.CLAHE(p=0.8),
            # albu.HorizontalFlip(p=0.15),

            # albu.VerticalFlip(p=0.5),
        ])
    if(phase == 'test'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    # list_transforms.extend([
    #     albu.Normalize(mean=(0.5, 0.5, 0.5),
    #                    std=(0.5, 0.5, 0.5), p=1),
    #     ToTensor()
    # ])
    if(phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(
            list_transforms,
            bbox_params=albu.BboxParams(
                format=bbox_format,
                min_area=min_area,
                min_visibility=min_visibility,
                label_fields=['category_id']))
            # keypoint_params=albu.KeypointParams(
            #     format='xy'))

def _crop_with_mks(image, boxes, labels, mks, img_dim, rgb_means, min_face_size):
    height, width, _ = image.shape

    for _ in range(250):
        if random.uniform(0, 1) <= 0.5:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.) # default setting for BI
            # scale = random.uniform(0.1, 1.) # bias toward big face setting (e.g. selfie)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 0.99999)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        mks_t = mks[mask_a].copy()

        # ignore tiny faces
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > min_face_size
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b].copy()
        mks_t = mks_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        mks_t[..., :2] -= roi[:2]

        return image_t, boxes_t, labels_t, mks_t

    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_means
    image_t[0:0 + height, 0:0 + width] = image

    return image_t, boxes, labels, mks

def _crop(image, boxes, labels, img_dim, rgb_means, min_face_size):
    height, width, _ = image.shape

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.) # default setting for BI
            # scale = random.uniform(0.1, 1.) # bias toward big face setting (e.g. selfie)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 0.99999)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        # ignore tiny faces
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > min_face_size
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        return image_t, boxes_t, labels_t

    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_means
    image_t[0:0 + height, 0:0 + width] = image

    return image_t, boxes, labels


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

def shuffleLR(x, match_parts):
    x_l = x
    x_r = x.copy()
    x_l[match_parts[:, 0]] = x_r[match_parts[:, 1]]
    x_l[match_parts[:, 1]] = x_r[match_parts[:, 0]]
    return x_l

def _mirror_with_mks(image, boxes, mks):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        mks[:, :, 0] = width - mks[:, :, 0]
        mks_mirror = []
        for item in mks:
            mks_mirror.append(shuffleLR(item, np.int32([[0, 1], [3, 4]])))
        mks = np.float32(mks_mirror)
    return image, boxes, mks

def _rotate_90_with_mks(image, boxes, mks):
    if random.randrange(3) == 0:
        h, w, _ = image.shape
        image = np.rot90(image)
        boxes_t = boxes.copy()
        boxes_t[:, 0], boxes_t[:, 1] = boxes[:, 1], w - boxes[:, 0]
        boxes_t[:, 2], boxes_t[:, 3] = boxes[:, 3], w - boxes[:, 2]
        mks_t = []
        for points in mks:
            mks_t.append([[y, w - x] for x, y in points])
        # import cv2
        # idx = random.randrange(20000)
        # cv2.imwrite("./test_aug/{:5}.jpg".format(idx), image)
        # image = cv2.imread("./test_aug/{:5}.jpg".format(idx))
        # from aitupu.common import image_trick
        # for points in mks_t:
        #     image_trick.draw_landmarks(image, points, r=5, color=(0, 0, 255))

        # for box in boxes_t:
        #     image_trick.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]),
        #         int(box[3])))
        # cv2.imwrite("./test_aug/{:5}.jpg".format(idx), image)

        mks_t = np.array(mks_t)
        return image, boxes_t, mks_t
    return image, boxes, mks


def preproc_for_test(image, insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    return image


class preproc(object):

    def __init__(self, img_dim, rgb_means, with_mks=False, with_vis=False,
            raw_box=False, min_face_size=24, to_gray_ratio=0.1):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.with_mks = with_mks
        self.with_vis = with_vis
        self.raw_box = raw_box
        self.min_face_size = min_face_size
        from torchvision.transforms import RandomGrayscale
        self.rgb_gray = RandomGrayscale    
        self.gray_ratio = to_gray_ratio 

    def __call__(self, image, targets):
        if self.with_mks:
            if self.with_vis:
                targets, mks, vis = targets
            else:
                targets, mks = targets
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # image_t = image
        image_t = self.rgb_gray(self.gray_ratio)(Image.fromarray(image))
        image_t = np.array(image_t)
        # image_t, boxes_t = _expand(image_t, boxes, self.cfg['rgb_mean'], self.cfg['max_expand_ratio'])
        if self.with_mks:
            if self.with_vis:
                mks = np.concatenate([mks, vis], 2)
            image_t, boxes_t, labels_t, mks_t = _crop_with_mks(image_t, boxes, labels, mks,
                                                               self.img_dim, self.rgb_means,
                                                               self.min_face_size)
            # image_t, boxes_t, mks_t = _rotate_90_with_mks(
            #         image_t, boxes_t, mks_t)
            image_t, boxes_t, mks_t = _mirror_with_mks(image_t, boxes_t, mks_t)
            if self.with_vis:
                vis_t = mks_t[:, :, 2:]
                mks_t = mks_t[:, :, :2]
        else:
            image_t, boxes_t, labels_t = _crop(image_t, boxes, labels,
                                               self.img_dim,
                                               self.rgb_means,
                                               self.min_face_size)
            image_t, boxes_t = _mirror(image_t, boxes_t)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.img_dim, self.rgb_means)
        image_t = _distort(image_t)
        if not self.raw_box:
            boxes_t[:, 0::2] /= width
            boxes_t[:, 1::2] /= height
            if self.with_mks:
                mks_t[:, :, 0] /= width
                mks_t[:, :, 1] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        if self.with_mks:
            if self.with_vis:
                return image_t, (targets_t, mks_t, vis_t)
            else:
                return image_t, (targets_t, mks_t)
        else:
            return image_t, targets_t
