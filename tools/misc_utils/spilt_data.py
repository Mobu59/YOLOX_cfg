import json
import random


#path = '/world/data-gpu-94/liyang/pedDetection/final_bi_train.json'
path = '/world/data-gpu-94/goods_detection_data/all_goods.json'
path = '/world/data-gpu-94/ped_detection_data/biped.v8.head.mix.shuf.json'
path = '/world/data-gpu-94/smart_shelf_data/renamed_gt.v1.json'
path = '/world/data-gpu-94/ped_detection_data/bi_headtop/export_data/data.json'
path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v2/data.json'
path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v3/data.json'
path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/data.json'
#path = '/world/data-gpu-94/liyang/pedDetection/head_detection/data.json'
with open(path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    train_list = []
    val_list = []
    test_list = []
    for i, line in enumerate(lines):
        #k, info = line.strip().split('\t')[0:2]
        line = line.strip().split('\t')
        if len(line) < 2 :
            continue
        k, info = line[0], line[1]
        if i <= 0.6 * len(lines):
            train_list.append((k, info))
        elif 0.6 * len(lines) < i <= 0.8 * len(lines):
            val_list.append((k,info))
        else:
            test_list.append((k, info))
#with open('/world/data-gpu-94/liyang/pedDetection/Bi/train1.json', 'w') as f:
#with open('/world/data-gpu-94/goods_detection_data/train.v5.json', 'w') as f:
#with open('/world/data-gpu-94/liyang/pedDetection/head_detection/bi_head_20220323/train.json', 'w') as f:
#with open('/world/data-gpu-94/smart_shelf_data/train.v1.json', 'w') as f:
with open('/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/train.v1.json', 'w') as f:
#with open('/world/data-gpu-94/liyang/pedDetection/head_detection/train.v2.json', 'w') as f:
    for i in train_list:
        f.write(i[0] + '\t' + i[1] + '\n')
#with open('/world/data-gpu-94/goods_detection_data/val.v2.json', 'w') as f:
#with open('/world/data-gpu-94/smart_shelf_data/val.v1.json', 'w') as f:
with open('/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/val.v1.json', 'w') as f:
    for i in val_list:
        f.write(i[0] + '\t' + i[1] + '\n')
#with open('/world/data-gpu-94/liyang/pedDetection/Bi/test1.json', 'w') as f:
#with open('/world/data-gpu-94/goods_detection_data/test.v3.json', 'w') as f:
#with open('/world/data-gpu-94/liyang/pedDetection/head_detection/bi_head_20220323/val.json', 'w') as f:
#with open('/world/data-gpu-94/smart_shelf_data/test.v1.json', 'w') as f:
#with open('/world/data-gpu-94/liyang/pedDetection/head_detection/val.v2.json', 'w') as f:
with open('/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/test.v1.json', 'w') as f:
    for i in test_list:
        f.write(i[0] + '\t' + i[1] + '\n')



