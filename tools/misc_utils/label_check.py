import cv2 
import json
import numpy as np
import random
import os


def check_face():
    data_path = '/world/data-gpu-94/face_detection_data/test_list.json'
    data_path = '/world/data-gpu-94/face_detection_data/po_data/po_data_test/po_data_test.json'
    data_path = '/world/data-gpu-94/face_detection_data/po_train.latest.json'
    data_path = '/world/data-gpu-94/face_detection_data/internet_data_test/internet_data.test.json'
    data_path = '/world/data-gpu-94/liyang/pedDetection/to_fix/bi_data/export_full.json'
    data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json'
    data_path = '/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/val.json'
    data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/dirty_data.json'
    data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/2000_test.json'
    data_path = '/world/data-gpu-94/goods_detection_data/train.v3.json'
    data_path = '/world/data-gpu-94/goods_detection_data/train.v4.json'
    data_path = '/world/data-gpu-94/goods_detection_data/test.v2.json'
    data_path = '/home/liyang/YOLOX/YOLOX_outputs/yolox_voc_s/eval/head1_detection.json'
    data_path = '/world/data-gpu-94/goods_detection_data/train.v5.json'
    data_path = '/world/data-gpu-94/goods_detection_data/train.v6_20220301.json'
    data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/train1.json'
    data_path = '/world/data-gpu-94/ped_detection_data/biped.v7.head.mix.shuf.json'
    data_path = '/world/data-gpu-94/liyang/pedDetection/head_detection/fixed_val.json'
    #data_path = '/world/data-gpu-94/goods_detection_data/full_pedestrian_fixed.json'
    save_path = '/world/data-gpu-94/liyang/testshow/'
    #save_path = '/world/data-gpu-94/liyang/pedDetection/Faceboxes_train_samples/'
    #black_gt = open('./black_gt.json', 'w')
    with open(data_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[0:1000]
        count = 0
        #name_lst = []
        lst = []
        
        for line in lines:
            flags = []
            line = line.strip().split("\t")
            k, info = line[:2]
            info = json.loads(info)
            img = cv2.imread(k)
            if img is None:
                continue
            for i in info:
                x0 = int(i['xmin'])
                y0 = int(i['ymin'])
                x1 = int(i['xmax'])
                y1 = int(i['ymax'])
                name = int(i['name'])
                #roi = img[y0:y1, x0:x1]
                #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #_, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                #pixel_0 = len(thresh.astype(np.int8)[thresh==0])
                #pixel_1 = len(thresh.astype(np.int8)[thresh==255])
                #if pixel_0 / (pixel_0 + pixel_1) >= 0.9:
                #    flags.append('1')
                #    x0, y0, x1, y1 = 0, 0, 0, 0
                image = cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
                #name_lst.append(name)
                #if name == 4:
                #    lst.append(k)
            #if '1' in flags:
            #    black_gt.write(k + '\t' + json.dumps(info) + '\n')
            #    print('find 1 img')
                cv2.imwrite(save_path+ '{}'.format(os.path.split(k)[-1]), image)
                count += 1
        print(count)    
        #print(set(name_lst))
        #print(lst)

def check_face_bi():
    data_path = '/world/data-gpu-94/face_detection_data/internet_data.test.baseline.json' 
    save_path = '/world/data-gpu-94/liyang/pedDetection/test_show/'
    with open(data_path, 'r') as f:
        lines = f.readlines()[0:1000]
        for line in lines:
            line = json.loads(line)
            k, info = line['file_name'], line['objects']
            image = cv2.imread(k)
            for i in info:
               x0 = int(i['bbox'][0]) 
               y0 = int(i['bbox'][1]) 
               x1 = int(i['bbox'][2]) 
               y1 = int(i['bbox'][3]) 
               cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.imwrite(save_path + '{}.jpg'.format(k.split("/")[-1]), image)   

def check_face_bi_train():
    data_path = '/world/data-gpu-94/face_detection_data/bi_internet_facemask_train.json' 
    save_path = '/world/data-gpu-94/liyang/pedDetection/test_show/'
    save_path = '/world/data-gpu-94/liyang/test_show/'
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            k, info = line[0], line[1]
            info = json.loads(info)
            name = k.split('/')[-1].split('.jpg')[0]
            if '.' in name or name.isdigit() == False:
                continue
            if 2000 < int(name) < 3000:
                image = cv2.imread(k)
                for i in info:
                   x0 = int(i['xmin']) 
                   y0 = int(i['ymin']) 
                   x1 = int(i['xmax']) 
                   y1 = int(i['ymax']) 
                   cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.imwrite(save_path + '{}.jpg'.format(k.split("/")[-1]), image)   

def check_gt_img():
    path = '/world/data-gpu-94/face_detection_data/bi_internet_facemask_train.json' 
    #path = '/home/liyang/project/facedet/scripts/evalv2/demo_dets.json'
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            k = line[0]
            info = json.loads(line[1])
            img_name = os.path.split(k)[0]
            if img_name == '/world/data-gpu-94/face_detection_data/bi_face_mask_data/data' and 1000 <= int(os.path.splitext(k.split('/')[-1])[0]) < 2000:
                img = cv2.imread(k)
                #for i in info:
                #    x0 = int(i['xmin'])
                #    y0 = int(i['ymin'])
                #    x1 = int(i['xmax'])
                #    y1 = int(i['ymax'])
                #    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/Faceboxes_train_samples/{}'.format(k.split('/')[-1]), img)    
                #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/test_show/{}'.format(k.split('/')[-1]), img)    

def clean_bi_train_data():
    fp_data = '/world/data-gpu-94/liyang/pedDetection/export_json/biface_fp.json'
    f_fp = open(fp_data, 'r')
    with open('/world/data-gpu-94/liyang/pedDetection/bi_face.json', 'w') as f:
        lines_gt = f_gt.readlines()
        list_gt = []
        list_fp = []
        for line in lines_gt:
            line = line.strip().split('\t')
            k, info = line[:2]
            info = json.loads(info)
            list_gt.append((k, info))
        print("parse gt data finished", len(list_gt))    
        lines_fp = f_fp.readlines()
        for line in lines_fp:
            line = line.strip().split('\t')
            k, info = line[:2]
            list_fp.append(k)
        print("parse fp data finished", len(list_fp))    
        for i in list_gt:
            if i[0] in list_fp:
                continue
            f.write(i[0] + '\t' + json.dumps(i[1]) + '\n')
    f_fp.close()   

def vis_gt_fp():
    fp_path = '/home/liyang/project/facedet/scripts/badcase/fp.json'
    f_fp = open(fp_path, 'r')
    lines = f_fp.readlines()
    lst = []
    for line in lines:
        line = line.strip().split('\t')
        k, infos = line[0], json.loads(line[1])
        lst.append(k)
        img = cv2.imread(k)
        for info in infos:
            xmin = int(info['xmin'])
            ymin = int(info['ymin'])
            xmax = int(info['xmax'])
            ymax = int(info['ymax'])
            score = float(info['score'])
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255))
            cv2.putText(img, str(score), (xmin, ymin-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 255, 255))
        cv2.imwrite('/world/data-gpu-94/liyang/testshow/{}'.format(os.path.split(k)[-1]), img)    
    f_fp.close()
def vis_gt():
    gt_data = '/world/data-gpu-94/face_detection_data/po_test.latest.json'
    f_gt = open(gt_data, 'r') 
    lines_gt = f_gt.readlines()
    lst = []
    for i in os.listdir('/world/data-gpu-94/liyang/testshow/'):
        lst.append(i)
    for line in lines_gt:
        line = line.strip().split('\t')
        k, infos = line[0], json.loads(line[1])
        if os.path.split(k)[-1] in lst:
            img = cv2.imread('/world/data-gpu-94/liyang/testshow/' + os.path.split(k)[-1])
            for info in infos:
                xmin = int(info['xmin'])
                ymin = int(info['ymin'])
                xmax = int(info['xmax'])
                ymax = int(info['ymax'])
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0))
            cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/test_show/{}.jpg'.format(os.path.split(k)[-1]), img)    
    f_gt.close()            
def vis_unique_img():
    path = '/world/data-gpu-94/face_detection_data/internet_data_test/data/1_0000050.jpg'
    path = '/world/data-gpu-94/face_detection_data/internet_data_test/data/1_0000236.jpg'
    path = '/world/data-gpu-94/face_detection_data/internet_data_test/data/1_0000340.jpg'
    path = '/world/data-gpu-94/face_detection_data/internet_data_test/data/1_0000955.jpg'
    path = '/world/data-gpu-94/face_detection_data/internet_data_test/data/1_0001142.jpg'
    path = '/world/data-gpu-94/face_detection_data/bi_data/data/world/data-gpu-57/Tranning-data/liuliu/20171122/liuliu_cats/14545076088300.9976358329877257.jpg'
    test_data = '/world/data-gpu-94/face_detection_data/po_test.latest.json'
    test_data = '/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json'
    with open(test_data, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            k, infos = line[0], json.loads(line[1])
            if k == path:
                img = cv2.imread(k)
                for info in infos:
                    xmin = int(info['xmin'])
                    ymin = int(info['ymin'])
                    xmax = int(info['xmax'])
                    ymax = int(info['ymax'])
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/test_show/{}.jpg'.format(os.path.split(path)[-1]), img)    

def pick_fp_fn():
    gt = '/world/data-gpu-94/face_detection_data/internet_data_test/internet_data.test.json'
    gt = '/world/data-gpu-94/goods_detection_data/test.v3.json'
    fp = '/home/liyang/project/facedet/scripts/badcase/fp.json'
    fn = '/home/liyang/project/facedet/scripts/badcase/fn.json'
    fixed_json = '/world/data-gpu-94/liyang/pedDetection/to_fix/bi_data/fixed.json'
    #with open('/home/liyang/project/facedet/scripts/badcase/to_fix.json', 'w') as f:
    #with open('/world/data-gpu-94/liyang/pedDetection/Bi/new_test.json', 'w') as f:
    with open('/world/data-gpu-94/goods_detection_data/export_full.json', 'w') as f:
        lst1 = []
        lst2 = []
        #f_fp = open(fp, 'r')
        f_fn = open(fn, 'r')
        #lines_fp = f_fp.readlines()
        lines_fn = f_fn.readlines()
        #lines = lines_fp + lines_fn
        #f_fix = open(fixed_json, 'r')
        #lines = f_fix.readlines()
        for line in lines_fn:
            line = line.strip().split('\t')
            k, info= line[0], json.loads(line[1])
            #lst1.append((k, info))
            lst2.append(k)
        f_gt = open(gt, 'r')    
        lines_gt = f_gt.readlines()
        for line in lines_gt:
            line = line.strip().split('\t')
            k, info = line[0], json.loads(line[1])
            #if os.path.join('/world/data-gpu-94/liyang/pedDetection/to_fix/bi_data', os.path.split(k)[-1]) in lst2:
            if k not in lst2:
                continue
            lst1.append((k, info))
        dic = {}
        for i in lst1:
            if i[0] not in dic:
                dic[i[0]] = []
            #i[1] = json.loads(i[1])
            for item in i[1]:
                x0 = item['xmin']
                y0 = item['ymin']
                x1 = item['xmax']
                y1 = item['ymax']
                name = int(item['name'])
                difficult = 0
                degree = item['degree']
                td = {'ymax':y1, 'xmax':x1, 'xmin':x0, 'ymin':y0, 'name':name,
                        'difficult':difficult, 'degree':degree}
                dic[i[0]].append(td)
        for i in dic:        
            f.write(i + '\t' + json.dumps(dic[i]) + '\n')
    #f_fp.close()        
    f_fn.close()        
    f_gt.close()        
    #f_fix.close()

def find_cat_dog():
    test_data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json'
    lst = []
    with open(test_data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            k, item = line[0], json.loads(line[1])
            for i in item:
                name = i['name']
                if int(name) != 0:
                    lst.append((k, item))
    with open('/world/data-gpu-94/liyang/pedDetection/Bi/dirty_data.json', 'w') as f:
        for i in lst:
            f.write(i[0] + '\t' + json.dumps(i[1]) + '\n')

def cat_abs_path():
    path = '/world/data-gpu-94/goods_detection_data/train.v3.json'
    with open(path, 'r') as f:
        lines = f.readlines()
        path_lst = [] 
        for line in lines:
            line = line.strip().split('\t')
            k = line[0]
            abs = os.path.split(k)[0]
            path_lst.append(abs)
        print(set(path_lst))    

def get_goods_per_category_count():
    path = '/world/data-gpu-94/goods_detection_data/train.v5.json'
    path = '/world/data-gpu-94/goods_detection_data/val.v2.json'
    path = '/world/data-gpu-94/goods_detection_data/test.v3.json'
    path = '/world/data-gpu-94/liyang/pedDetection/head_detection/fixed_train.json'
    with open(path, 'r') as f:
        lines = f.readlines()
        category = {}
        for line in lines:
            line = line.strip().split('\t')
            k, item = line[0], json.loads(line[1])
            for i in item:
                name = str(i['name'])
                if name not in category:
                    category[name] = []
                category[name].append(k)    
    for i in category:
        print('category {}, count is {}'.format(i, len(category[i])))


if __name__ == "__main__":
    #check_face_bi_train()
    #check_gt_img()
    #check_face()
    #clean_bi_train_data()
    #vis_unique_img()
    #vis_gt_fp()
    #vis_gt()
    #pick_fp_fn()
    #find_cat_dog()
    #cat_abs_path()
    get_goods_per_category_count()

