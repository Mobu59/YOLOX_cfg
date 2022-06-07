import os 
import cv2
def rename_img():
    path = '/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/JPEGImages'
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        k = i.split('/')[-1]
        name = os.path.splitext(k)[0]
        if ' ' in name:
            name = name.replace(' ', '')
        cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/jpegs/{}.jpg'.format(name), img)    

def rename_json():
    path = '/home/liyang/project/facedet/scripts/badcase/to_fix.json'
    path = '/world/data-gpu-94/liyang/pedDetection/Bi/new_test.json'
    path = '/home/liyang/YOLOX/YOLOX_outputs/yolox_voc_s/eval/head1_detection.json'
    path = '/world/data-gpu-94/goods_detection_data/watson_5422/fixed_watson.json'
    with open(path, 'r') as f:
        lines = f.readlines()
        lst = []
        for line in lines:
            line = line.strip().split('\t')
            k, info = line[:2]
            #k = '/world/data-gpu-94/liyang/pedDetection/to_fix/bi_data/' + os.path.split(k)[-1]
            #k = '/world/data-gpu-94/face_detection_data/internet_data_test/data/' + os.path.split(k)[-1]
            k = '/world/data-gpu-94/goods_detection_data/watson_5422/' + os.path.split(k)[-1]
            lst.append((k, info))
    #with open('/world/data-gpu-94/liyang/pedDetection/to_fix/bi_data/export_full.json', 'w') as f:
    #with open('/world/data-gpu-94/liyang/pedDetection/Bi/new_test.v1.json', 'w') as f:
    with open('/world/data-gpu-94/goods_detection_data/watson.json', 'w') as f:
        for i in lst:
            f.write(i[0] + '\t' + i[1] + '\n')
rename_json()            
