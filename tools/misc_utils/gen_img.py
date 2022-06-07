import os
import cv2
import xml.etree.ElementTree as et

def read_xml(abs_path,xml_id,ori_img_path,count):
    in_file=os.path.join(abs_path,xml_id)
    tree=et.parse(in_file)
    root=tree.getroot()
    filename=root.find("filename").text
    save_path="/world/data-gpu-94/liyang/pedDetection/splitdata/ped_trainimgs"
    for i in os.listdir(ori_img_path):
        if i==filename:
            print("{} img get".format(count))
            img=cv2.imread(os.path.join(ori_img_path,i))
            cv2.imwrite(os.path.join(save_path,i),img)
def get_img():
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            k = line[0]
            abs = k.split('/')[-1]
            save_name = os.path.splitext(abs)[0] 
            if ' ' in save_name:
                save_name = save_name.replace(' ', '')
            img = cv2.imread(k)
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/JPEGImages/{}.jpg'.format(save_name), img)
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/to_fix/bi_data/{}.jpg'.format(save_name), img)
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/ahs/VOC2007/JPEGImages/{}.jpg'.format(save_name), img)
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/head/VOC2007/JPEGImages/{}.jpg'.format(save_name), img)
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/shelf/VOC2007/JPEGImages/{}.jpg'.format(save_name), img)
            #cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/ped/VOC2007/JPEGImages/{}.jpg'.format(save_name), img)
            cv2.imwrite('/world/data-gpu-94/liyang/pedDetection/data/vertical_ped_v4/VOC2007/JPEGImages/{}.jpg'.format(save_name), img)
if __name__ == "__main__":
    path="/world/data-gpu-94/liyang/pedDetection/splitdata/pedtrainxml"
    path = '/world/data-gpu-94/face_detection_data/bi_internet_facemask_train.json'
    path = '/world/data-gpu-94/face_detection_data/po_data/data_20211207.json'
    path = '/world/data-gpu-94/face_detection_data/bi_face_mask_data/bi_face_mask_data.json'
    path = '/home/liyang/project/facedet/scripts/badcase/to_fix.json'
    path = '/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/val.json'
    path = '/world/data-gpu-94/goods_detection_data/val.v2.json'
    path = '/world/data-gpu-94/goods_detection_data/val.v3_20220301.json'
    path = '/world/data-gpu-94/liyang/pedDetection/head_detection/bi_head_20220323/val.json'
    path = '/world/data-gpu-94/liyang/pedDetection/head_detection/fixed_val.json'
    path = '/world/data-gpu-94/smart_shelf_data/data_v1/train.v1.json'
    path = '/world/data-gpu-94/smart_shelf_data/data_v2/val.json'
    path = '/world/data-gpu-94/ped_detection_data/bi_headtop/export_data/test.v1.json'
    path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/val.v1.json'
    #path = '/world/data-gpu-94/liyang/pedDetection/head_detection/val.v2.json'
    get_img()

