# -*- coding: utf-8 -*-
import os 
import cv2
import json
from tqdm import tqdm
from xml.dom.minidom import Document


class CreateAnno():
    def __init__(self,):
        self.doc=Document()
        self.anno = self.doc.createElement('annotation')  # 创建根元素
        self.doc.appendChild(self.anno)
        self.add_folder()
        self.add_path()
        self.add_source()
        self.add_segmented()
    def add_folder(self, floder_text_str='JPEGImages'):
        floder = self.doc.createElement('floder')  ##建立自己的开头
        floder_text = self.doc.createTextNode(floder_text_str)
        ##建立自己的文本信息
        floder.appendChild(floder_text)  ##自己的内容
        self.anno.appendChild(floder)
    def add_filename(self, filename_text_str='00000.jpg'):
        filename = self.doc.createElement('filename')
        filename_text = self.doc.createTextNode(filename_text_str)
        filename.appendChild(filename_text)
        self.anno.appendChild(filename)
    def add_path(self, path_text_str="None"):
        path = self.doc.createElement('path')
        path_text = self.doc.createTextNode(path_text_str)
        path.appendChild(path_text)
        self.anno.appendChild(path)
    def add_source(self, database_text_str="Unknow"):
        source = self.doc.createElement('source')
        database = self.doc.createElement('database')
        database_text = self.doc.createTextNode(database_text_str)  # 元素内容写入
        database.appendChild(database_text)
        source.appendChild(database)
        self.anno.appendChild(source)
    def add_pic_size(self, width_text_str="0", height_text_str="0",
        depth_text_str="3"):
        size = self.doc.createElement('size')
        width = self.doc.createElement('width')
        width_text = self.doc.createTextNode(width_text_str)  # 元素内容写入
        width.appendChild(width_text)
        size.appendChild(width)

        height = self.doc.createElement('height')
        height_text = self.doc.createTextNode(height_text_str)
        height.appendChild(height_text)
        size.appendChild(height)
                                 
        depth = self.doc.createElement('depth')
        depth_text = self.doc.createTextNode(depth_text_str)
        depth.appendChild(depth_text)
        size.appendChild(depth)
        self.anno.appendChild(size)
    def add_segmented(self, segmented_text_str="0"):
        segmented = self.doc.createElement('segmented')
        segmented_text = self.doc.createTextNode(segmented_text_str)
        segmented.appendChild(segmented_text)
        self.anno.appendChild(segmented)
    def add_object(self, name_text_str="None", xmin_text_str="0", ymin_text_str="0", xmax_text_str="0", ymax_text_str="0", pose_text_str="Unspecified", truncated_text_str="0", difficult_text_str="0"):
        object = self.doc.createElement('object')
        name = self.doc.createElement('name')
        name_text = self.doc.createTextNode(name_text_str)
        name.appendChild(name_text)
        object.appendChild(name)

        pose = self.doc.createElement('pose')
        pose_text = self.doc.createTextNode(pose_text_str)
        pose.appendChild(pose_text)
        object.appendChild(pose)

        truncated = self.doc.createElement('truncated')
        truncated_text = self.doc.createTextNode(truncated_text_str)
        truncated.appendChild(truncated_text)
        object.appendChild(truncated)

        difficult = self.doc.createElement('difficult')
        difficult_text = self.doc.createTextNode(difficult_text_str)
        difficult.appendChild(difficult_text)
        object.appendChild(difficult)

        bndbox = self.doc.createElement('bndbox')
        xmin = self.doc.createElement('xmin')
        xmin_text = self.doc.createTextNode(xmin_text_str)
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = self.doc.createElement('ymin')
        ymin_text = self.doc.createTextNode(ymin_text_str)
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = self.doc.createElement('xmax')
        xmax_text = self.doc.createTextNode(xmax_text_str)
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = self.doc.createElement('ymax')
        ymax_text = self.doc.createTextNode(ymax_text_str)
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)
        self.anno.appendChild(object)
    def get_anno(self):
        return self.anno
    def get_doc(self):
        return self.doc
    def save_doc(self, save_path):
        with open(save_path, "w")as f:
            self.doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
def json_to_xml(json_path, save_path):
    json_path = line
    save_path = xml_path
    k = json_path[0]
    h, w, c = cv2.imread(k).shape
    file_name = k.split("/")[-1]
    if ' ' in file_name:
        file_name = file_name.replace(' ', '')
    filename = os.path.splitext(file_name)[0]
    #if filename.endswith(".jpeg"):
    #    filename = filename.replace(".jpeg", ".jpg")
    #xml_name = filename.replace(".jpg", ".xml")
    xml_name = filename + '.xml'
    info = json.loads(json_path[1])
    info_lst = []
    for i in info:
        cls, x1, y1, x2, y2 = i["name"], i["xmin"], i["ymin"], i["xmax"], i["ymax"]
        info_lst.append([x1, y1, x2, y2, cls])
    xml_anno = CreateAnno()
    xml_anno.add_filename(file_name)
    xml_anno.add_pic_size(width_text_str=str(h),
            height_text_str=str(w), depth_text_str=str(3))
    for x1, y1, x2, y2, cls in info_lst:
        xml_anno.add_object(name_text_str=str(cls),
                            xmin_text_str=str(int(x1)),
                            ymin_text_str=str(int(y1)),
                            xmax_text_str=str(int(x2)),
                            ymax_text_str=str(int(y2)))
    xml_anno.save_doc(save_path + xml_name)
if __name__ == "__main__":
    #path = "/world/data-gpu-94/liyang/pedDetection/bi_test.json"
    path = '/world/data-gpu-94/face_detection_data/bi_internet_facemask_train.json'
    path = '/world/data-gpu-94/face_detection_data/po_data/data_20211207.json'
    path = '/world/data-gpu-94/face_detection_data/bi_face_mask_data/bi_face_mask_data.json'
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
#    xml_path = "/world/data-gpu-94/liyang/pedDetection/splitdata/pedtestxml/"
    xml_path = "/world/data-gpu-94/liyang/pedDetection/data/VOCdevkit/VOC2007/annos/"
    xml_path = "/world/data-gpu-94/liyang/pedDetection/data/ahs/VOC2007/Annotations/"
    xml_path = "/world/data-gpu-94/liyang/pedDetection/data/goods/VOC2007/Annotations/"
    xml_path = "/world/data-gpu-94/liyang/pedDetection/data/head/VOC2007/Annotations/"
    #xml_path = "/world/data-gpu-94/liyang/pedDetection/data/shelf/VOC2007/Annotations/"
    #xml_path = "/world/data-gpu-94/liyang/pedDetection/data/ped/VOC2007/Annotations/"
    xml_path = "/world/data-gpu-94/liyang/pedDetection/data/vertical_ped_v4/VOC2007/Annotations/"
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            line = line.strip().split("\t")
            json_to_xml(line, xml_path)
