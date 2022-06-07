import cv2
import json

data_path = '/world/data-gpu-94/goods_detection_data/train.v5.json'
data_path = '/world/data-gpu-94/goods_detection_data/test.v3.json'
data_path = '/world/data-gpu-94/goods_detection_data/val.v2.json'
dirty_data = open('./val.v2.dirty_data.txt', 'w')
with open(data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        count = 0
        lst = []
        line = line.strip().split('\t')
        k, items = line[0], json.loads(line[1])
        if 'shelf' in k or 'wly' in k or 'watson' in k:
            continue
        img = cv2.imread(k)
        h, w = img.shape[0], img.shape[1]
        #cv2.line(img, (0, int(h * 1 / 6)), (w, int(h * 1 / 6)), (0, 0, 255), 2)
        save_name = k.split('/')[-1]
        #cv2.imwrite('/world/data-gpu-94/liyang/testshow/' + save_name, img)
        for item in items:
            x0, y0, x1, y1 = item['xmin'], item['ymin'], item['xmax'], item['ymax']
            #cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        #cv2.imwrite('/world/data-gpu-94/liyang/testshow/' + save_name, img)
            lst.append(y0)
        for i in lst:
            if i < (h * 1 / 6):
            #if y0 < (h * 6 / 25) and loop >= 35:
                count += 1
        if count <= 6:        
            #cv2.imwrite('/home/liyang/YOLOX/tools/show/' + save_name, img)
            dirty_data.write(k + '\n')
dirty_data.close()            

