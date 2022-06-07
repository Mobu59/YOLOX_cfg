import cv2
import json
#all_img = '/world/data-gpu-94/goods_detection_data/train.v5.json'
all_img = '/world/data-gpu-94/goods_detection_data/test.v3.json'
all_img = '/world/data-gpu-94/goods_detection_data/val.v2.json'
#dirty_img = '/home/liyang/YOLOX/tools/find_goods_dirty_data/train.v5.dirty_data.txt'
dirty_img = '/home/liyang/YOLOX/tools/find_goods_dirty_data/test.v3.dirty_data.txt'
dirty_img = '/home/liyang/YOLOX/tools/find_goods_dirty_data/val.v2.dirty_data.txt'
#save_path = '/world/data-gpu-94/goods_detection_data/train.v6_20220301.json'
save_path = '/world/data-gpu-94/goods_detection_data/test.v4_20220301.json'
save_path = '/world/data-gpu-94/goods_detection_data/val.v3_20220301.json'
keep = []
dirty = open(dirty_img, 'r')
save_json = open(save_path, 'w')
dirty_lines = dirty.readlines()
lst = []
for line in dirty_lines:
    line = line.split('\n')[0]
    lst.append(line)
with open(all_img, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        k, items = line[0], json.loads(line[1])
        if k in lst:
            continue
        keep.append((k, items))
for i in keep:
    save_json.write(i[0] + '\t' + json.dumps(i[1]) + '\n')
            #img = cv2.imread(k)
            #save_name = k.split('/')[-1]
            #for i in items:
            #    x0, y0, x1, y1 = int(i['xmin']), int(i['ymin']), int(i['xmax']), int(i['ymax'])
            #    cv2.line(img, (0, int(img.shape[0]*1/6)), (img.shape[1], int(img.shape[0]*1/6)), (0, 0, 255), 2)
            #    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
            #cv2.imwrite('/world/data-gpu-94/liyang/testshow/' + save_name, img)    

                
