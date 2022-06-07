import os
import random


#train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/head_detection/train/anno'
#train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/Annotations'
train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/annos'
txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/ImageSets/Main/'
train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/ahs/VOC2007/Annotations'
txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/ahs/VOC2007/ImageSets/Main/'
train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/goods/VOC2007/Annotations'
txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/goods/VOC2007/ImageSets/Main/'
train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/head/VOC2007/Annotations'
txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/head/VOC2007/ImageSets/Main/'
train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/vertical_ped_v4/VOC2007/Annotations'
txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/vertical_ped_v4/VOC2007/ImageSets/Main/'
#train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/shelf/VOC2007/Annotations'
#txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/shelf/VOC2007/ImageSets/Main/'
#train_xml_filepath = '/world/data-gpu-94/liyang/pedDetection/data/ped/VOC2007/Annotations'
#txt_savepath = '/world/data-gpu-94/liyang/pedDetection/data/ped/VOC2007/ImageSets/Main/'
#trainval_per = 0.1
#train_per = 0.9
#num = len(os.listdir(train_xml_filepath))
#lst = range(num)
#tv = int(num * trainval_per)
#tr = int(tv * train_per)
#trainval = random.sample(lst, tv)
#train = random.sample(trainval, tr)
#ftest = open((txt_savepath + 'val.txt'), 'w') 
#ftrain = open((txt_savepath + 'trainval.txt'), 'w')  
#ftest = open((txt_savepath + 'aa.txt'), 'w') 
#ftrain = open((txt_savepath + 'bb.txt'), 'w')  
f = open((txt_savepath + 'val.txt'), 'w')  
for i in os.listdir(train_xml_filepath):
    name = i[:-4] + '\n'
    f.write(name)
#lst = []
#for i in os.listdir(train_xml_filepath):
#    name = i[:-4] + '\n'
#    lst.append(name)
#random.shuffle(lst)    
#count = 0
#for i in lst:
#    if count <= 0.9 *len(lst):
##    name = os.listdir(train_xml_filepath)[i][:-4] + '\n'
##    if i in trainval:
##        ftest.write(name)
##    else:
##        ftrain.write(name)
#        ftrain.write(i)
#        count += 1
#    if count > 0.9 * len(lst): 
#        ftest.write(i)
#ftest.close()
#ftrain.close()
f.close()
