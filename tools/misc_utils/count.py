import os
count=0
path1 = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/Annotations'
path2 = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/test' 
path3 = '/world/data-gpu-94/liyang/pedDetection/Faceboxes_train_samples' 
path4 = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/annos' 
path4 = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/JPEGImages'
path4 = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/xml_dir'
for i in os.listdir(path4):
    count+=1
print(count)
