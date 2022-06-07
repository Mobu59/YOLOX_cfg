#!/usr/bin/env python3
import os
import os.path
import random
import pickle
import xml.etree.ElementTree as ET
from loguru import logger

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval ,voc_eval_v1

from .datasets_wrapper import Dataset
from .voc_classes import VOC_CLASSES
from config import *
import sys

#cfg = get_cfg("goods_det")
cfg = get_cfg(sys.argv[2])

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
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        #self._classes = 1
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
        img, label = self._parse_line(index)
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

        # print("tpdataset pull item", target)
        return img, target, img_info, index

    # def load_anno(self, index):
    #     return self.annotations[index][0]

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

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

    def _parse_line(self, idx):
        k, info = self.ids[idx]
        res = []
        mkss = []
        viss = []
        img = cv2.imread(k, cv2.IMREAD_COLOR)
        if img is None:
            print("path is ", k)
        h, w, _ = img.shape
        for face_info in info:
            xmin = max(0, face_info['xmin'])
            xmax = min(w, face_info['xmax'])
            ymin = max(0, face_info['ymin'])
            ymax = min(h, face_info['ymax'])
            name = face_info['name']
            #if xmin > w or ymin > h or xmax < 0 or ymax < 0 or int(name) == 1:
            if xmin > w or ymin > h or xmax < 0 or ymax < 0:
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
            if name == cfg["ignore_label"]:
                continue

            #name = 1 - name

            # if 'difficult' in face_info.keys() and face_info['difficult'] == 2:
            #     name = self.difficult_label

            res.append([xmin, ymin, xmax, ymax, name])
            # res.append([xmin, ymin, bbox_w, bbox_h, name])

        # r = min(self.img_size[0] / h, self.img_size[1] / w)
        if len(res) == 0:
            res = [[0, 0, 0, 0, 0]]
        if cfg["task_name"] == "goods_det":    
            prob = np.random.randint(0, 2)
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
        res = np.float32(res)
        return img, res
