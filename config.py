#config.py
"""
In your case, you should modify "python tools/train.pt -task ***" where *** is one of (goods_det, ahs_det, ver_ped_det, head_det, hands_goods_det, sens_det_yolox_l, sens_det_yolox_m)
"""
import random

#垂直角度行人检测
cfg_ver_ped_det = {
        "task_name": "ver_ped_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.33,
        "width": 0.375,
        "warmup_epochs": 2,
        "no_aug_epochs": 100,
        "max_epoch": 100,
        "flip_prob": 0.5,
        "degrees": 90.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (288, 288),
        "test_size": (288, 512),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/ver_ped_det",
        #"train_data_dir": "/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/train.v1.json",
        "train_data_dir": "/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/data.json",
        "val_data_dir": "/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/val.v1.json",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_tiny.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":3.4,
        "fill_value":random.randint(0, 255)
}

#head检测
cfg_head_det_tiny = {
        "task_name": "head_det_tiny",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.33,
        "width": 0.375,
        "warmup_epochs": 2,
        "no_aug_epochs": 50,
        "max_epoch": 50,
        "flip_prob": 0.5,
        "degrees": 0.373,
        #"degrees": 30.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        #"input_size": (288, 288),
        "input_size": (416, 416),
        #"test_size": (288, 512),
        "test_size": (576, 1024),
        #"test_size": (512, 896),
        #"test_size": (448, 768),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        #"lr": 0.01 / 128.0,
        "lr": 0.01 / 64.0,
        "eval_interval": 100,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/head_det_tiny",
        #"train_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/train.v2.json",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/new_train.v1.json",
        "val_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/val.v2.json",
        "gpu_num": 4,
        #"batch_size": 128,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_tiny.pth',
        #"pretrain_model": '/world/data-gpu-94/liyang/cfg_yolox/trained_models/head_det_tiny/yolox_all_base/latest_ckpt.pth',
        "max_labels":100,
        "ignore_label":None,
        "aspect_ratio":2.4,
        "fill_value":random.randint(0, 255)
}

#行人检测
cfg_ped_det_tiny = {
        "task_name": "ped_det_tiny",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.33,
        "width": 0.375,
        "warmup_epochs": 2,
        "no_aug_epochs": 10,
        "max_epoch": 50,
        "flip_prob": 0.5,
        #"degrees": 0.373,
        "degrees": 30.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (288, 288),
        "test_size": (288, 512),
        #"test_size": (256, 448),
        #"test_size": (224, 384),
        #"test_size": (576, 1024),
        #"test_size": (512, 896),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        #"lr": 0.01 / 128.0,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/ped_det_tiny",
        "train_data_dir": "/world/data-gpu-94/liyang/ped_det/biped.ped.train.v3.json",
        "val_data_dir": "/world/data-gpu-94/liyang/ped_det/biped.ped.val.v3.json",
        "gpu_num": 4,
        #"batch_size": 128,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_tiny.pth',
        "max_labels":50,
        "ignore_label":1,
        "aspect_ratio":6.4,
        "fill_value":random.randint(0, 255)
}        

cfg_head_det_nano = {
        "task_name": "head_det_nano",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.33,
        "width": 0.25,
        "warmup_epochs": 2,
        "no_aug_epochs": 50,
        "max_epoch": 50,
        "flip_prob": 0.5,
        "degrees": 0.373,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        #"input_size": (288, 288),
        "input_size": (416, 416),
        "test_size": (288, 512),
        #"test_size": (256, 448),
        #"test_size": (224, 384),
        #"test_size": (576, 1024),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        #"lr": 0.01 / 128.0,
        "lr": 0.01 / 64.0,
        "eval_interval": 100,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/head_det_nano",
        #"train_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/train.v2.json",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/new_train.v1.json",
        "val_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/val.v2.json",
        "gpu_num": 2,
        #"batch_size": 128,
        "batch_size": 64,
        #"pretrain_model": '/home/liyang/cfg_yolox/pretrain_weights/yolox_nano.pth',
        "pretrain_model": None,
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4,
        "fill_value":random.randint(0, 255)
}

#Goods detection
cfg_goods_det = {
        "task_name": "goods_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0", "1", "2", "3", "4", "5", "6", "7", "8"),
        "num_classes": 9,
        "depth": 0.67,
        "width": 0.75,
        "warmup_epochs": 1,
        "no_aug_epochs": 20,
        "max_epoch": 20,
        "flip_prob": 0.5,
        "degrees": 0.373,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (416, 416),
        "test_size": (704, 416),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 32.0,
        "eval_interval": 1,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/goods_det",
        #"train_data_dir": "/world/data-gpu-94/goods_detection_data/train.v6_20220301.json",
        #"train_data_dir": "/world/data-gpu-94/goods_detection_data/train.v8.json",#添加了单行数据和badcase
        #"train_data_dir": "/world/data-gpu-94/goods_detection_data/train.v9.json",#v9将屈臣氏中人工标注的数据拿出来和五粮液的结合
        #"train_data_dir": "/world/data-gpu-94/goods_detection_data/singleRowShelfData.json",#单行数据
        "train_data_dir": "/world/data-gpu-94/goods_detection_data/train.v10.json",#v10是v9和单行数据的结合
        "val_data_dir": "/world/data-gpu-94/goods_detection_data/val.v3_20220301.json",
        "gpu_num": 4,
        "batch_size": 32,
        #"pretrain_model": 'pretrain_weights/yolox_m.pth',
        "pretrain_model": '/world/data-gpu-94/liyang/cfg_yolox/trained_models/goods_det/yolox_all/using_weights/latest_ckpt.pth',
        "max_labels":200,
        "ignore_label":5,
        "aspect_ratio":10,
        "fill_value":random.randint(0, 255)
}

#手拿商品检测
cfg_hands_goods_det = {
        "task_name": "hands_goods_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0", "1", "2"),
        "num_classes": 3,
        "depth": 0.67,
        "width": 0.75,
        "warmup_epochs": 0,
        "no_aug_epochs": 100,
        "max_epoch": 100,
        "flip_prob": 0.5,
        #"degrees": 0.373,
        "degrees": 1.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (416, 416),
        "test_size": (416, 736),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 100,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/hands_goods_det",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v2/train.json",
        #"val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v2/val.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.json",
        #"val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/val.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.v2.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.v3.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v4/train.v4.json",
        "train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.v5.json",
        "val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v4/test.v2.json",
        "gpu_num": 4,
        "batch_size": 64,
        #"pretrain_model": "/world/data-gpu-94/wyq/cfg_yolox/trained_models/hands_goods_det_v3/yolox_all/epoch_73_ckpt.pth",
        #"pretrain_model": "/world/data-gpu-94/wyq/cfg_yolox/trained_models/hands_goods_det_v3/yolox_all/epoch_99_ckpt.pth",
        "pretrain_model": "/home/liyang/cfg_yolox/pretrain_weights/yolox_m.pth",
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":10.0,
        "fill_value":random.randint(0, 255)
}

#手拿商品检测
cfg_hands_goods_x_det = {
        "task_name": "hands_goods_x_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0", "1"),
        "num_classes": 2,
        "depth": 1.0,
        "width": 1.0,
        "warmup_epochs": 1,
        "no_aug_epochs": 20,
        "max_epoch": 20,
        "flip_prob": 0.5,
        #"degrees": 0.373,
        #"degrees": 180.0,
        "degrees": 30.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (640, 640),
        "test_size": (640, 640),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "/world/data-gpu-94/wyq/cfg_yolox/trained_models/hands_goods_det",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v2/train.json",
        #"val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v2/val.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.json",
        #"val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/val.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.v2.json",
        #"train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v3/train.v3.json",
        "train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v4/train.v1.json",
        "val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v4/test.v1.json",
        "gpu_num": 2,
        "batch_size": 8,
        "pretrain_model": None,
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":10.0,
        "fill_value":random.randint(0, 255)
}

#敏感人物yolox_m
cfg_sens_det_yolox_m = {
        "task_name": "sens_det_yolox_m",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.67,
        "width": 0.75,
        "warmup_epochs": 2,
        "no_aug_epochs": 20,
        "max_epoch": 20,
        "flip_prob": 0.5,
        "degrees": 180.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (416, 416),
        "test_size": (416, 416),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 100,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/sens_det_yolox_m",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json",
        "val_data_dir": "/world/data-gpu-94/liyang/pedDetection/Bi/val.json",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_m.pth',
        "max_labels":100,
        "ignore_label":None,
        "aspect_ratio":2.4,
        "fill_value":random.randint(0, 255)
}

#爱回收YOLOX_M
cfg_ahs_det = {
        "task_name": "ahs_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0", "1", "2"),
        "num_classes": 3,
        "depth": 0.67,
        "width": 0.75,
        "warmup_epochs": 2,
        "no_aug_epochs": 20,
        "max_epoch": 20,
        "flip_prob": 0.5,
        "degrees": 0.373,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (416, 416),
        "test_size": (416, 416),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "/world/data-gpu-94/liyang/cfg_yolox/trained_models/ahs_det",
        "train_data_dir": "/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/train.json",
        "val_data_dir": "/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/val.json",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_m.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4,
        "fill_value":random.randint(0, 255)
}

#敏感人物YOLOX_L版本
cfg_sens_det_yolox_l = {
        "task_name": "sens_det_yolox_l",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 1.00,
        "width": 1.00,
        "warmup_epochs": 2,
        "no_aug_epochs": 12,
        "max_epoch": 12,
        "flip_prob": 0.5,
        "degrees": 90.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (640, 640),
        "test_size": (640, 640),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "./trained_models/sens_det_yolox_l",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json",
        "val_data_dir": "None",
        "gpu_num": 4,
        "batch_size": 8,
        "pretrain_model": 'pretrain_weights/yolox_l.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4,
        "fill_value":random.randint(0, 255)
}

#To be continued


cfg_zoo = {
    cfg_ver_ped_det["task_name"]: cfg_ver_ped_det,
    cfg_head_det_tiny["task_name"]: cfg_head_det_tiny,
    cfg_head_det_nano["task_name"]: cfg_head_det_nano,
    cfg_goods_det["task_name"]: cfg_goods_det,
    cfg_hands_goods_det["task_name"]: cfg_hands_goods_det,
    cfg_hands_goods_x_det["task_name"]: cfg_hands_goods_x_det,
    cfg_sens_det_yolox_m["task_name"]: cfg_sens_det_yolox_m,
    cfg_ahs_det["task_name"]: cfg_ahs_det,
    cfg_sens_det_yolox_l["task_name"]: cfg_sens_det_yolox_l,
    cfg_ped_det_tiny["task_name"]: cfg_ped_det_tiny,
}

def get_cfg(name):
    return cfg_zoo[name]

if __name__ == "__main__":
    cfg = get_cfg("goods_det")
    cfg = get_cfg("head_det_tiny")
    cfg = get_cfg("head_det_nano")
    print(cfg)
    print(cfg['width'])
    #print(cfg_zoo.keys())

