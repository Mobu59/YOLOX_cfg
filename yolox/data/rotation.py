import cv2
import json
import imutils
import numpy as np
import math
import random
import cv2
import numpy as np

from .data_augment import get_aug_params, random_affine
from config import *
import sys

cfg = get_cfg(sys.argv[2])
fill_value = cfg['fill_value']
def rotate_bound(img, angle, borderValue=(114,114,114)):
    h, w = img.shape[:2]
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += nW / 2 - cX
    M[1, 2] += nH / 2 - cY
    new_img = cv2.warpAffine(img, M, (nW, nH), borderValue=borderValue)

    return new_img

def Srotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex - pointx) * math.cos(angle) + (valuey - pointy) * math.sin(angle) + pointx
    sRotatey = (valuey - pointy) * math.cos(angle) - (valuex - pointx) * math.sin(angle) + pointy
    return sRotatex, sRotatey

def ellipserotate(angle, b_H, b_W, x_c, y_c, pointx, pointy):
    y_1 = []
    x_1 = np.arange(x_c - b_W / 2, x_c + b_W / 2, 0.1)
    for x in x_1:
        y = math.sqrt(math.fabs((1 -(x - x_c) ** 2 / (b_W ** 2 / 4)) * (b_H
            ** 2) / 4)) + y_c
        y_1.append(y)
    for x in x_1:
        y = -math.sqrt(math.fabs((1 - (x - x_c) ** 2 / (b_W ** 2 / 4)) * (b_H
            ** 2) / 4)) + y_c
        y_1.append(y)
    x_1 = np.append(x_1, x_1)
    y_1 = np.array(y_1)
    x_2, y_2 = Srotate(math.radians(angle), x_1, y_1, pointx, pointy)

    #assert x_2.size != 0
    #assert y_2.size != 0
    if x_2.size == 0:
        x_2 = np.zeros((490,))
    if y_2.size == 0:
        y_2 = np.zeros((490,))

    x_min = np.min(x_2)
    y_min = np.min(y_2)
    x_max = np.max(x_2)
    y_max = np.max(y_2)

    return x_min, y_min, x_max, y_max

def getRotatedImg(img, label, degrees):
    angle = get_aug_params(degrees)
    #rotation_image = imutils.rotate_bound(img, angle)
    rotation_image = rotate_bound(img, angle, borderValue=(fill_value,fill_value,fill_value))
    height, width = img.shape[:2]
    height_new, width_new = rotation_image.shape[:2]
    i = label.copy()
    for j in range(i.shape[0]):
        x_c = i[j, 0] + (i[j, 2] - i[j, 0]) / 2
        y_c = i[j, 1] + (i[j, 3] - i[j, 1]) / 2
        b_W = i[j, 2] - i[j, 0]
        b_H = i[j, 3] - i[j, 1]
                
        pointx = width/2
        pointy = height/2

        x_min, y_min, x_max, y_max = ellipserotate(-angle, 
                b_H,
                b_W,
                x_c,
                y_c,
                pointx,
                pointy
        )
        x_min = x_min + (width_new - width) / 2
        x_max = x_max + (width_new - width) / 2
        y_min = y_min + (height_new - height) / 2
        y_max = y_max + (height_new - height) / 2

        label[j, 0] = x_min 
        label[j, 1] = y_min  
        label[j, 2] = x_max  
        label[j, 3] = y_max  

    return rotation_image, label

def vis():
    with open('/home/liyang/Data-Augmentation/ellipse_rotation/test/3.txt', 'r') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            line = line.strip().split('\t')
            img_path ,item = line[0], json.loads(line[1])
            #img = cv2.imread(img_path)
            img = cv2.imread('/home/liyang/Data-Augmentation/ellipse_rotation/test/3.jpg')
            h, w, _ = img.shape
            for i in item:
                xc = float(i['xc']) 
                yc = float(i['yc']) 
                w_ = float(i['w']) 
                h_ = float(i['h']) 
                x0 = int(xc - w_ / 2)
                y0 = int(yc - h_ / 2)
                x1 = int(xc + w_ / 2)
                y1 = int(yc + h_ / 2)
                print((x0,y0, x1, y1))
                cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 1)
            cv2.imwrite('/home/liyang/Data-Augmentation/ellipse_rotation/test/{}.jpg'.format(j+100), img)    

if __name__ == '__main__':
    angle = 15
    #img_path = "/home/liyang/Data-Augmentation/ellipse_rotation/test/0000122_01200_d_0000119.jpg"
    #txt_path = "/home/liyang/Data-Augmentation/ellipse_rotation/test/0000122_01200_d_0000119.txt"
    path = '/world/data-gpu-94/liyang/pedDetection/Bi/2000_test.json'

    img_write_path = '/home/liyang/Data-Augmentation/ellipse_rotation/test/3.jpg'
    outpath_re = "/home/liyang/Data-Augmentation/ellipse_rotation/test/3.txt"
    getRotatedImg(angle, img_write_path, path)
    vis()
