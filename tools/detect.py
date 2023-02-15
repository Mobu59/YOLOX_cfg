#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
import random
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.data.datasets import VOC_CLASSES, GOODS_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("-task", "--task_name", type=str, default="goods_det", help="goods_det, ahs_det, sens_det_yolox_m, ver_ped_det, head_det, hands_goods_det, sens_det_yolox_l")
    parser.add_argument(
       '-do', "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path",
        default="/home/liyang/YOLOX/1.png", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument(
        "--save_as_video",
        action="store_true",
        help="whether to save the inference result of image as video",
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='/home/liyang/YOLOX/exps/example/yolox_voc/yolox_voc_s.py',
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt",
            default='/home/liyang/YOLOX/YOLOX_outputs/yolox_voc_s/best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.35, type=float, help="test conf")
    parser.add_argument("--nms", default=0.5, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            path = img
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        if img is not None:
            height, width = img.shape[:2]
            img_info["height"] = height
            img_info["width"] = width
            img_info["raw_img"] = img

            #ratio = min(288 / img.shape[0], 512 / img.shape[1])
            #self.test_size = (288, 512)
            #self.test_size = (416, 416)
            ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
            img_info["ratio"] = ratio
            img, _ = self.preproc(img, None, self.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.float()
            if self.device == "gpu":
                img = img.cuda()
                if self.fp16:
                    img = img.half()  # to FP16

            with torch.no_grad():
                t0 = time.time()
                outputs = self.model(img)
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            return outputs, img_info
        else:
            return [None, None]

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        #scores = output[:, 4] * output[:, 5]
        scores = output[:, 4]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result, f, save_as_video, vid_writer):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    thresh = 800
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        #if outputs is None or outputs[0] is None:
        #    f.write('{:s} {}'.format(image_name, '[]') + '\n')
        if outputs is None or outputs[0] is None: 
            f.write('{:s} {}'.format(image_name, '[]') + '\n')
            if save_as_video:
                result_image = cv2.imread(image_name)
                vid_writer.write(result_image)
            if save_result:
                result_image = cv2.imread(image_name)
                if result_image is None:
                    continue
                h, w, _ = result_image.shape
                if min(w, h) >= thresh:
                    r = min(thresh / w, thresh / h) 
                    result_image = cv2.resize(result_image, (int(w * r), int(h * r)))
                save_folder = os.path.join(
                    vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                )
                os.makedirs(save_folder, exist_ok=True)
                save_name = image_name.split('/')[-2] + "_" + image_name.split('/')[-1]
                save_file_name = os.path.join(save_folder, save_name)
                save_file_name = save_file_name + '.jpg' 
                logger.info("Saving detection result in {}".format(save_file_name))
                cv2.imwrite(save_file_name, result_image)
            continue
        ratio = img_info['ratio']
        for k in range(outputs[0].shape[0]):
            x0 = outputs[0][k, 0].item() / ratio
            y0 = outputs[0][k, 1].item() / ratio
            x1 = outputs[0][k, 2].item() / ratio
            y1 = outputs[0][k, 3].item() / ratio
            obj_conf = outputs[0][k, 4].item()
            cls_conf = outputs[0][k, 5].item()
            cls = outputs[0][k, 6].item()
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {}'.format(image_name, (obj_conf * cls_conf), x0, y0, x1,
                        y1, int(cls)) + '\n')
            #f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {}'.format(image_name, obj_conf, x0, y0, x1, y1, cls) + '\n')
        if save_result:
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            h, w, _ = result_image.shape
            if min(w, h) >= thresh:
                r = min(thresh / w, thresh / h) 
                result_image = cv2.resize(result_image, (int(w * r), int(h * r)))
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_name = image_name.split('/')[-2] + "_" + image_name.split('/')[-1]
            save_file_name = os.path.join(save_folder, save_name)
            save_file_name = save_file_name + '.jpg' 
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        #ch = cv2.waitKey(0)
        #if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #    break
        
        #将视频帧推理得到的结果保存成视频
        if save_as_video:
            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            vid_writer.write(result_image)

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    s_ratio = fps / 10.0
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps/s_ratio, (int(width), int(height))
    )
    cnt = 1
    #while True:
    while cnt <= 4000:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            #ch = cv2.waitKey(1)
            #if ch == 27 or ch == ord("q") or ch == ord("Q"):
            #    break
        else:
            break
        cnt += 1        


def main(exp, args, test_data, save_path):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    # whether to save img to a directory
    #args.save_result = True

    vis_folder = None
    if args.save_result or args.save_as_video:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
       
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        #trt_file = os.path.join(file_name, "model_trt.pth")
        trt_file = '/home/liyang/cfg_yolox/goods_det/yolox_all/model_trt.pth'
        trt_file = '/world/data-gpu-94/liyang/cfg_yolox/trained_models/hands_goods_det/yolox_all/model_trt.pth'
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        #model, exp, COCO_CLASSES, trt_file, decoder,
        model, exp, VOC_CLASSES, trt_file, decoder,
        #model, exp, GOODS_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        #test_data_path = '/world/data-gpu-94/liyang/pedDetection/to_fix/test_head.json'   
        #test_data_path = '/world/data-gpu-94/liyang/pedDetection/bi_test.json'   
        #test_data_path = '/world/data-gpu-94/face_detection_data/po_test.latest.json'   
        if test_data_path.endswith(('.json', '.txt')):
            with open(test_data_path, 'r') as f:
                img_cnt = 0
                lines = f.readlines()
                vid_writer = None
                if args.save_as_video:
                    args.path = lines[0].strip().split("\t")[0]
                    img = cv2.imread(args.path)
                    height, width = img.shape[:2]
                    save_folder = os.path.join(
                        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                    )
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, args.path.split("/")[-2].split("_imgs")[0] + ".mp4")
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (int(width), int(height))
                    )
                #random.shuffle(lines)
                #lines = lines[:300]
                for line in lines:
                    img_cnt += 1
                    line = line.strip().split('\t')
                    #line = line.strip().split('\n')
                    k = line[0]
                    args.path = k
                    image_demo(predictor, vis_folder, args.path, current_time, args.save_result, save_txt, args.save_as_video, vid_writer)
                print("-----------------------------------------")    
                print("{} lines have been parsed".format(img_cnt))    
                print("-----------------------------------------")    
            save_txt.close()            
        elif test_data_path.endswith(('.png', '.jpg')):
            args.path = test_data_path 
            #image_demo(predictor, vis_folder, args.path, current_time, args.save_result, None)
            image_demo(predictor, vis_folder, args.path, current_time,
                    args.save_result, save_txt, save_as_video, vid_writer)
    elif args.demo == "video" or args.demo == "webcam":
        dir = True
        if dir:
            video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/test_videso'
            video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/20220810_test_videos'
            #video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/xlsx_files/20220830_测试视频'
            video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/20220906_test_videos'
            video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230131_videos'
            video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos'
            #video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/20220715_videos'
            #video_dir = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/20220729_videos'
            video_lst = os.listdir(video_dir)
            #random.shuffle(video_lst)
            for cnt, i in enumerate(video_lst):
                args.path = os.path.join(video_dir, i)
                if os.path.isfile(args.path):
                    imageflow_demo(predictor, vis_folder, current_time, args)
                    if cnt == 7:
                        exit()
        else:            
            imageflow_demo(predictor, vis_folder, current_time, args)
    
if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    args = make_parser().parse_args()
    #743张政治人物测试图
    #test_data_path = '/world/data-gpu-94/face_detection_data/po_data/po_data_test/po_data_test.json'

    #test_data_path = '/world/data-gpu-94/face_detection_data/po_test.latest.json'   
    #test_data_path = '/world/data-gpu-94/face_detection_data/internet_data_test/internet_data.test.json'   
    #test_data_path = '/world/data-gpu-94/face_detection_data/internet_data_test/data/1_0016391.jpg'
    #政治人物测试集
    #test_data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/new_test.v1.json'   

    #test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v2/test.json'   
    #test_data_path = '/world/data-gpu-94/ped_detection_data/bi_headtop/export_data/to_infer_data/foreign_mexico.json'
    #test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v1/test.v1.json'   
    #test_data_path = '/world/data-gpu-94/goods_detection_data/test.v3.json'
    test_data_path = "/world/data-gpu-94/goods_detection_data/tupu_retrain_data.json"
    #test_data_path = '/world/data-gpu-94/liyang/test.json'
    #test_data_path = "/world/data-gpu-94/goods_detection_data/singleRowShelfData/39-A4FC2-D-1-1.jpg"
    #test_data_path = '/world/data-gpu-94/liyang/full_pedestrian/test.json'
    #test_data_path = '/home/liyang/YOLOX/test_images/1.jpg'
    #test_data_path = '/home/liyang/YOLOX/test_images/test_imgs.json'
    #test_data_path = '/world/data-gpu-94/ped_detection_data/biped_data/part2/data/00381604_15236814955900.7714032098837007.jpg'
    #test_data_path = '/world/data-gpu-94/ped_detection_data/biped.v7.head.mix.shuf.test.json'
    test_data_path = '/world/data-gpu-94/liyang/ped_det/biped.ped.test.json'
    test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v3/test.json'
    test_data_path = '/world/data-gpu-94/smart_shelf_data/20220708_data.json'
    test_data_path = '/world/data-gpu-94/liyang/ped_det/biped.ped.test.v3.json'
    test_data_path = '/world/data-gpu-94/smart_shelf_data/20220715_0729_data.json'
    test_data_path = '/world/data-gpu-94/sku_data/tupu/tupu.txt'
    test_data_path = '/world/data-gpu-94/sku_data/tupu/gz/gz.txt'
    test_data_path = '/world/data-gpu-94/sku_data/tupu/hz_p2/hz_p2.txt'
    test_data_path = '/world/data-gpu-94/goods_detection_data/badcase/20220804_wtscg.json'
    test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v3/test.v4.json'
    #test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v3/data.v5.json'
    test_data_path = '/world/data-gpu-94/ped_detection_data/biped.v8.head.mix.shuf.test.json'
    test_data_path = '/world/data-gpu-94/ped_detection_data/bi_headtop/daily_headtop_data/test_data.json'
    test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v4/test.v2.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_302097_imgs.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_imgs.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_imgs_20230202.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_imgs_20230207.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_imgs_20230201_02.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_imgs_20230208.json'
    test_data_path = '/world/data-gpu-94/liyang/shelf_test_imgs/shelf_videos/smart_shelf_videos/20230201_videos/Fps10_test_imgs_0208.json'
    #test_data_path = '/world/data-gpu-94/smart_shelf_data/20220901_data.json'
    #test_data_path = '/world/data-gpu-94/wyq/mobile_video_data/demo2020/hw_ped/ped3/frame_000650.jpg'
    #test_data_path = '/world/data-gpu-94/wyq/mobile_video_data/demo2020/hw_ped/ped3.txt'
    #test_data_path = '/world/data-gpu-94/liyang/misc_projects/star_card_debug.json'
    #test_data_path = '/world/data-gpu-94/goods_detection_data/badcase/20220804_wtscg/test.json'
    #test_data_path = '/world/data-gpu-94/goods_detection_data/test.v4_20220301.json'
    #test_data_path = '/world/data-gpu-94/star_card/stardata/stardata.txt'
    #test_data_path = '/world/data-gpu-94/star_card/stardata/stardata.shuf.2w.txt'
    #test_data_path = '/world/data-c4/wash/2022-08-09/5eddb86afaf52c4ca704ea41/data.txt'
    #test_data_path = '/world/data-gpu-94/liyang/goods_retrain_data/badcase.json'
    #test_data_path = '/home/liyang/cfg_yolox/tmp.jpg'
    #test_data_path = '/world/data-gpu-94/wyq/mobile_video_data/demo2020/huigou/pass1.txt'
    #test_data_path = '/world/data-gpu-94/smart_shelf_data/data_v3/test.json'
    #test_data_path = '/world/data-gpu-94/smart_shelf_data/20220704_data.json'
    #args.path = '/world/data-gpu-94/wyq/mobile_video_data/demo2020/pass/过店东.mp4'
    #test_data_path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v2/test.v2.json'
    #test_data_path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/test.v1.json'
    #test_data_path = '/world/data-gpu-94/liyang/test.json'
    #test_data_path = '/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_p1_test.json'
    #test_data_path = '/world/data-gpu-94/slam/demo/tracking/camera1.json'
    #test_data_path = '/home/liyang/YOLOX/tools/misc_utils/find_goods_dirty_data/test.v3.dirty_data.txt'
    #test_data_path = '/world/data-gpu-94/sku_data/watsons_db_images/watsons_db_images.json'
    #test_data_path = '/world/data-gpu-94/liyang/aihuishou_train/113_ahs_test.json'
    #test_data_path = '/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/test.json'   
    #test_data_path = '/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/rotated_test_img.json'   
    #test_data_path = '/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/rotated_90_test_img.json'   
    #test_data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/1000_test.json'   
    #test_data_path = '/world/data-gpu-94/liyang/aihuishou_train/ahs_url/ahs_test_imgs/ahs_test_imgs.json'   
    #test_data_path = '/world/data-gpu-94/liyang/pedDetection/Bi/dog_cat_test.json'
    args.path = '/world/data-gpu-94/wyq/mobile_video_data/demo2020/hw_ped/ped3.mp4'
    #args.path = '/world/data-gpu-94/wyq/mobile_video_data/demo2020/huigou/pass3.mp4'
    #args.path = '/home/liyang/cfg_yolox/865678.mp4'
    save_txt = open('/home/liyang/cfg_yolox/eval/test_f.txt', 'w')
    #save_txt = None
    exp = get_exp(args.exp_file, args.name)

    main(exp, args, test_data_path, save_txt)
