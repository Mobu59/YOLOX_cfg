import os
import cv2
import json
import tqdm
import torch
import random
import argparse
import numpy as np


def random_flip_horizontal(img, box, p=0.5):
    if np.random.random() < p:
        w = img.shape[1]
        img = img[:, ::-1, :]
        box[:, [0, 2]] = w - box[:, [2, 0]]
    return img, box    


def Large_Scale_Jittering(img, box, min_scale=0.5, max_scale=2.0):
    fill_value = np.random.randint(0, 255)
    ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    h_, w_ = int(h * ratio), int(w * ratio)
    img = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_LINEAR)

    return img, box
    #x, y = int(np.random.uniform(0, abs(w_ - w))), int(np.random.uniform(0, abs(h_ - h)))
    #if ratio <= 1.0:
    #    img_pad = np.zeros((h, w, 3), dtype=np.uint8) + fill_value
    #    img_pad[y:y + h_, x:x + w_, :] = img
    #    #box[:, [0, 2]] = box[:, [0, 2]] * w_ / w + x
    #    #box[:, [1, 3]] = box[:, [1, 3]] * h_ / h + y
    #    box[[0, 2]] = box[[0, 2]] * w_ / w + x
    #    box[[1, 3]] = box[[1, 3]] * h_ / h + y
    #    return img_pad, box
    #else:
    #    img_crop = img[y:y + h, x:x + w, :]
    #    #box[:, [0, 2]] = box[:, [0, 2]] * w_ / w - x
    #    #box[:, [1, 3]] = box[:, [1, 3]] * h_ / h - y
    #    box[[0, 2]] = box[[0, 2]] * w_ / w - x
    #    box[[1, 3]] = box[[1, 3]] * h_ / h - y
    #    return img_crop, box


#def img_add(img_src, img_main, mask_src, box_src):
#    h, w, c = img_main.shape
#    src_h, src_w = img_src.shape[0:2]
#    mask = np.array(mask_src, dtype=np.uint8)
#    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
#    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
#    mask_02 = np.array(mask_02, dtype=np.uint8)
#    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8), mask=mask_02)
#    
#    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (w, h), interpolation=cv2.INTER_NEAREST)
#    box_src[:, [0, 2]] = box_src[:, [0, 2]] * w / src_w 
#    box_src[:, [1, 3]] = box_src[:, [1, 3]] * h / src_h 
#
#    return img_main, box_src


def cal_iou(bbox1, bbox2):
    bbox1 = torch.Tensor(bbox1)
    bbox2 = torch.Tensor(bbox2)
    tl = torch.max(bbox1[:, None, :2], bbox2[:, :2])
    br= torch.min(bbox1[:, None, 2:], bbox2[:, 2:])
    area_a = torch.prod(bbox1[:, 2:] - bbox1[:, :2], 1)
    area_b = torch.prod(bbox2[:, 2:] - bbox2[:, :2], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en
    return area_i / (area_a[:, None] + area_b - area_i)

def copy_paste(img_main, img_src, label_main, label_src):

    save = False
    #img_main, box_main = random_flip_horizontal(img_main, label_main)
    img_main, box_main = img_main, label_main
    img_src, box_src = random_flip_horizontal(img_src, label_src)
    #img_main, box_main = Large_Scale_Jittering(img_main, box_main)
    #img_src, box_src = Large_Scale_Jittering(img_src, box_src)
    #随机缩放
    x0, y0, x1, y1 = int(box_main[0]), int(box_main[1]), int(box_main[2]), int(box_main[3])
    crop_img = img_main[y0:(y1 + 1), x0:(x1 + 1)]
    scale_crop_img, _ = Large_Scale_Jittering(crop_img, box_main)
    #随机粘贴
    h, w = scale_crop_img.shape[0:2]
    if img_src.shape[1] > w and img_src.shape[0] > h:
        pos = (np.random.randint(0, img_src.shape[1] - w), np.random.randint(0, img_src.shape[0] - h))
        src_h, src_w = img_src.shape[0:2]
        img_src[pos[1]:pos[1] + h, pos[0]:pos[0] + w] = scale_crop_img
        new_w = pos[0] + w    
        new_h = pos[1] + h    
        if new_w > img_src.shape[1]:
            new_w = img_src.shape[1] - 1
        if new_h > img_src.shape[0]:
            new_h = img_src.shape[0] + 1
        new_box = np.asarray([[pos[0], pos[1], new_w, new_h]])    
        box = np.vstack((box_src, new_box))
        iou = cal_iou(box_src, new_box)
        if float(torch.max(iou)) < 0.05:
            print(float(torch.max(iou)))
            save = True
        return img_src, box, save
    else:
        return img_src, np.array([[0,0,0,0]]), save


def main(args):
    with open("/world/data-gpu-94/smart_shelf_data/data_v3/train.v2.json", "r") as f:
        lines = f.readlines()[0:300]
        info = {}
        for line in lines:
            line = line.strip().split("\t")
            k, items = line[0], json.loads(line[1])
            if k not in info:
                info[k] = []
            for i in items:
                x0 = i['xmin']
                y0 = i['ymin']
                x1 = i['xmax']
                y1 = i['ymax']
                info[k].append([x0, y0, x1, y1])
    os.makedirs(args.output_dir, exist_ok=True)
    cnt = 0
    for src_name, bbox_list in info.items():
        imgs_list = list(info.keys())
        bbox_list = np.array(bbox_list, dtype=np.float32)
        id1 = np.random.randint(0, len(imgs_list))
        main_name = imgs_list[id1]
        main_label = np.array(info[main_name])
        if main_label.shape[0] == 1:
            main_label = main_label[0]
        else:
            id_ = np.random.randint(0, main_label.shape[0])
            main_label = main_label[id_]
        img_src = cv2.imread(src_name)
        img_main = cv2.imread(main_name)
        h, w, _ = img_src.shape
        if h < main_label[3] - main_label[1] or w < main_label[2] - main_label[0]:
            continue
        img, box, save = copy_paste(img_main, img_src, main_label, bbox_list, args.coincide, args.muti_obj)
        img = img.copy()
        for i in box:
            x0 = int(i[0])
            y0 = int(i[1])
            x1 = int(i[2])
            y1 = int(i[3])
            cv2.rectangle(img, (x0, y0), (x1, y1), 255, 2)
        if save:    
            cv2.imwrite(f"./xixixi/cpAug_jpg/{cnt}.jpg", img)    
            cnt += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./output_dir", type=str,
                        help="保存数据增强结果的路径")
    parser.add_argument("--coincide", default=False, type=bool,
                        help="True表示允许数据增强后的图像目标出现重合，默认不允许重合")
    parser.add_argument("--muti_obj", default=False, type=bool,
                        help="True表示将src图上的所有目标都复制粘贴，False表示只随机粘贴一个目标")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
