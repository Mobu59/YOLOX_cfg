#!/usr/bin/env python3
import os
import os.path
import random
import pickle
import cv2
import numpy as np
import albumentations as albu
import xml.etree.ElementTree as ET
from loguru import logger

import cv2
import copy
import numpy as np

from yolox.data import copy_paste, motion_blur
from yolox.evaluators.voc_eval import voc_eval ,voc_eval_v1

from .datasets_wrapper import Dataset
from .voc_classes import VOC_CLASSES
from config import *
import sys

#cfg = get_cfg("goods_det")
cfg = get_cfg(sys.argv[2])
fill_value = cfg['fill_value']

class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info

def get_po_faceboxes_augumentation(phase, width=640, height=640, min_area=0.,
        min_visibility=0., bbox_format='pascal_voc'):
    print("<<<<<<<<< using get_po_faceboxes_augumentation")
    list_transforms = []
    if phase == 'train':
        list_transforms.extend([
            albu.augmentations.transforms.JpegCompression(
                quality_lower=50, quality_upper=90,
                always_apply=False, p=0.5),
            albu.OneOf([
                albu.augmentations.transforms.RandomScale(
                    scale_limit=(0.5, 2), p=1.0),
                albu.augmentations.transforms.RandomScale(
                    scale_limit=(0.5, 2), p=1.0),
                albu.augmentations.transforms.RandomScale(
                    scale_limit=(0.5, 2), p=1.0),
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
                limit=20, interpolation=1, border_mode=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
                mask_value=[0, 0, 0], always_apply=False, p=0.7),
            # # albu.augmentations.transforms.Flip(),
            # # albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.2,
                                              contrast_limit=0.2),
                # albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            # albu.augmentations.transforms.Blur(blur_limit=14, p=0.5),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=5, b_shift_limit=5,
                              g_shift_limit=5),
                albu.HueSaturationValue(hue_shift_limit=10,
                                        sat_shift_limit=10),
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


class TPDataset(Dataset):
    def __init__(
        self,
        data_dir,
        img_size=(640, 640),
        preproc=None,
        target_transform=AnnotationTransform(),
        dataset_name="TPDataset",
        cache=False):

        super().__init__(img_size)
        self.root = data_dir
        self.img_size = img_size
        aug = get_po_faceboxes_augumentation(
                'train',
                width=img_size[0], height=img_size[1],
                min_area=100., min_visibility=0.7,
                bbox_format='pascal_voc')
        self.preproc = aug
        self.target_transform = target_transform
        self.name = dataset_name
        self._classes = VOC_CLASSES 
        self.ids = self._parse_dataset(data_dir)
        self.samples_n = None

        self.imgs = None

    def __len__(self):
        return len(self.ids)

    def load_resized_img(self, index):
        img, label = self.load_image_v2(index)
        h, w, _ = img.shape
        resized_img = np.zeros((640, 640, 3), dtype = np.uint8)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        try:
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
        except cv2.error as e:
            pass
        label[:, :4] = label[:, :4] * r

        return label, resized_img, (h, w)

    def load_image(self, index):
        img, _ = self._parse_line(index)
        assert img is not None
        return img

    def load_image_v2(self, index):
        img = None 
        while img is None:
            img, label = self._parse_line(index)
            index += 1
        assert img is not None
        return img, label

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        target, img, img_info = self.load_resized_img(index)
            # target, img_info, _ = self.annotations[index]
        aug_target = [] 
        if self.preproc is not None:
            bbox_index = [i for i in range(target.shape[0])]
            bbox_index = np.array(bbox_index, dtype=np.int8).reshape((len(bbox_index),))
            anno = {'image': img, 'bboxes': target[:, :4], 'category_id': target[:, 4]}
            try:
                t = self.preproc(**anno)
                if (len(t['bboxes']) == 0):
                    aug_target.append([0, 0, 0, 0, 0])
                else:
                    for bbox_idx, bbox in enumerate(t['bboxes']):
                        # bbox_list = [v/self.img_size[0] for v in bbox]
                        bbox_list = [v for v in bbox]
                        bbox_list.append(int(t['category_id'][bbox_idx]))
                        aug_target.append(bbox_list)
                img = t['image']
                target = np.array(aug_target, dtype=np.float)
            except Exception as e:
                # aug_target.append([0, 0, 0, 0, 0])
                print(e)
                k, info = self.ids[index]
                print("path", k)

        return img, target, img_info, index

    # def load_anno(self, index):
    #     return self.annotations[index][0]

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        aug_target = [] 
        if self.preproc is not None:
            bbox_index = [i for i in range(target.shape[0])]
            bbox_index = np.array(bbox_index, dtype=np.int8).reshape((len(bbox_index),))
            anno = {'image': img, 'bboxes': target[:, :4], 'category_id': target[:, 4]}
            t = self.preproc(**anno)
            if (len(t['bboxes']) == 0):
                aug_target.append([0, 0, 0, 0, 0])
            else:
                for bbox_idx, bbox in enumerate(t['bboxes']):
                    bbox_list = [v/self.img_size[0] for v in bbox]
                    bbox_list.append(t['category_id'][bbox_idx])
                    aug_target.append(bbox_list)

            img = t['image']
            target = np.array(aug_target)
        # print("tpdataset", target)
        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)

        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(os.path.splitext(self.root)[0], 'Bi', "results", "VOC" + 'yolox' , "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            #print("Writing {} VOC results file".format('face'))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    #index = index[1]
                    index = index[0]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            #"{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                            "{} {} {} {} {} {}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )
    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = os.path.join(os.path.splitext(self.root)[0], 'Bi', "VOC" + 'yolox')
        #name = self.image_set[0][1]
        annopath = os.path.join(rootpath, "Annotations", "{}.xml")
        #imagesetfile = os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
        imagesetfile = self.root
        cachedir = os.path.join(
            os.path.splitext(self.root)[0], 'Bi', "annotations_cache", "VOC" +
            'yolox', 'voc2007' 
            #'yolox', 'val' 
        )
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        # The PASCAL VOC metric changed in 2010
        #use_07_metric = True if int(self._year) < 2010 else False
        use_07_metric = True 
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(VOC_CLASSES):

            if cls == "__background__":
                continue

            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval_v1(
                filename,
                annopath,
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)
                        
    def _parse_dataset(self, root):
        import json
        import tqdm
        lines = open(root, 'r').readlines()
        # if self.samples_n is not None:
        #     print('samples_n', self.samples_n)
        #     lines = lines[:self.samples_n]
        datalines = []
        print('xxxx Parsing Dataset ...')
        for line in tqdm.tqdm(lines):
            try:
                items = line.strip().split('\t')
                if len(items) < 2:
                    continue
                k = items[0]
                info = items[1]
                info = json.loads(info)
                datalines.append((k, info))
            except Exception as e:
                print(line)
                print(e)
        return datalines

    def get_index(self, array):
        ind = np.random.randint(0, array.shape[0])
        y1 = array[ind][3]
        new_arr = np.where(np.abs(array[:, 3]-y1)<=30, 0, array[:, 3])
        index = np.nonzero(new_arr)
        zero_ind = np.where(new_arr==0)
        return index, zero_ind

    def _parse_line(self, idx):
        k, info = self.ids[idx]
        res = []
        mkss = []
        viss = []
        img = cv2.imread(k, cv2.IMREAD_COLOR)
        if img is None:
            print("path is ", k)
        #if "RPC" in k:
        #    img = motion_blur(img)
        date = k.split("/")[-2]
        #if "smart_shelf_data" in k and "0715" not in date and "2019" not in date:
        #    h, w, _ = img.shape
        #    img[0:int(h*0.38), 0:w] = fill_value
        #if "20220715" in date:
        #    h, w, _ = img.shape
        #    img[0:int(h*0.455), 0:w] = fill_value
        ori_img = img.copy()
        h, w, _ = img.shape
        for face_info in info:
            xmin = max(0, face_info['xmin'])
            xmax = min(w, face_info['xmax'])
            ymin = max(0, face_info['ymin'])
            ymax = min(h, face_info['ymax'])
            name = face_info['name']
            #由于调整了摄像头，导致比例不一定适配所有的图片，会有把标签框一起遮盖了的情况
            #if "20220715_0729" in date and ymin <= int(h * 0.455):
            #    continue
            #if xmin > w or ymin > h or xmax < 0 or ymax < 0 or int(name) == 1:
            if xmin > w or ymin > h or xmax > w or ymax > h or xmax < 0 or ymax < 0 or xmax - xmin <= 0 or ymax - ymin <= 0:
                #print(k, face_info)
                continue

            bbox_h = ymax - ymin
            bbox_w = xmax - xmin
            max_s = max(bbox_h, bbox_w)
            min_s = min(bbox_h, bbox_w)
            #try:
            #    if min_s/max_s < 0.3:
            #        continue
            #except ZeroDivisionError as e:
            #    pass

            name = int(name)

            #if name < 0 or name >0:
            if name < 0:
                name = 0
            if name == cfg["ignore_label"] or name == 6 or int(name) == 1:
                continue

            #name = 1 - name

            # if 'difficult' in face_info.keys() and face_info['difficult'] == 2:
            #     name = self.difficult_label

            res.append([xmin, ymin, xmax, ymax, name])
            # res.append([xmin, ymin, bbox_w, bbox_h, name])

        # r = min(self.img_size[0] / h, self.img_size[1] / w)
        if len(res) == 0:
            return None, None
        #if cfg["task_name"] == "hands_goods_det":
        #    prob = np.random.random()
        #    if prob > 0.4:
        #        idx_ = np.random.randint(0, len(self.ids)) 
        #        k_, info_ = self.ids[idx_]
        #        if "RPC" not in k_:
        #            img_ = cv2.imread(k_, cv2.IMREAD_COLOR)
        #            label = []
        #            h_, w_, _ = img_.shape
        #            for i in info_:
        #                xmin = max(0, i['xmin'])
        #                xmax = min(w_, i['xmax'])
        #                ymin = max(0, i['ymin'])
        #                ymax = min(h_, i['ymax'])
        #                name = i['name']
        #                label.append([xmin, ymin, xmax, ymax, name])
        #            if label == [[0, 0, 0, 0, 0]]:    
        #                img = ori_img
        #                res = res
        #            else:    
        #                if len(label) > 1:
        #                    random_id = np.random.randint(0, len(label))
        #                    main_label = np.array(label[random_id])
        #                else:
        #                    main_label = np.array(label[0])
        #                new_img, box, save = copy_paste(img_, img, main_label, np.array(res))    
        #                if save:
        #                    img = new_img
        #                    res = box
        #                else:
        #                    img = ori_img
        #                    res = res
        #        else:
        #            img = ori_img
        #            res = res

        if cfg["task_name"] == "goods_det":    
            prob = np.random.randint(0, 3)
            #随机涂抹黑框
            if prob == 1:
                if len(res) >= 10:
                    for i in range(int(1/3*len(res))):
                        count = np.random.randint(0, high=int(len(res)))
                        x0, y0, x1, y1 = res[count][:4]
                        x0 = int(x0)
                        y0 = int(y0)
                        x1 = int(x1)
                        y1 = int(y1)
                        img[y0:y1, x0:x1] = 0
                        res[count] = [0, 0, 0, 0, 0]
            #随机按行涂抹            
            elif prob == 2:
                res = np.asarray(res)
                labels = copy.deepcopy(res)
                if res.shape[0] > 0 and labels.shape[0] > 0:
                    index, zero_ind = self.get_index(res)
                    res = res[index]
                    labels = labels[zero_ind]
                else:
                    res = res
                for i in range(res.shape[0]):
                    if res.shape[0] == 1:
                        continue
                    x0 = int(res[i][0])
                    y0 = int(res[i][1])
                    x1 = int(res[i][2])
                    y1 = int(res[i][3])
                    img[y0:y1, x0:x1] = 0
                res = labels    
        res = np.float32(res)
        return img, res
