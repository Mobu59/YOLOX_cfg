#config.py
"""
In your case, you should modify "python tools/train.pt -task ***" where *** is one of (goods_det, ahs_det, ver_ped_det, head_det, hands_goods_det, sens_det_yolox_l, sens_det_yolox_m)
"""

#垂直角度行人检测
cfg_ver_ped_det = {
        "task_name": "ver_ped_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.33,
        "width": 0.375,
        "warmup_epochs": 5,
        "no_aug_epochs": 50,
        "max_epoch": 50,
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
        "output_dir": "./ver_ped_det",
        "train_data_dir": "/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/train.v1.json",
        "val_data_dir": "/world/data-gpu-94/liyang/vertical_ped_detection/export_data/data_v4/val.v1.json",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_tiny.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":3.4
}

#head检测
cfg_head_det = {
        "task_name": "head_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0",),
        "num_classes": 1,
        "depth": 0.33,
        "width": 0.375,
        "warmup_epochs": 2,
        "no_aug_epochs": 100,
        "max_epoch": 100,
        "flip_prob": 0.5,
        "degrees": 15.0,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (288, 288),
        "test_size": (288, 512),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 128.0,
        "eval_interval": 1,
        "output_dir": "./head_det",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/train.v2.json",
        "val_data_dir": "/world/data-gpu-94/liyang/pedDetection/head_detection/val.v2.json",
        "gpu_num": 4,
        "batch_size": 128,
        "pretrain_model": 'pretrain_weights/yolox_tiny.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4
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
        "test_size": (416, 416),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "./goods_det",
        #"train_data_dir": "/world/data-gpu-94/goods_detection_data/train.v6_20220301.json",
        "train_data_dir": "/world/data-gpu-94/goods_detection_data/train.v7.json",
        "val_data_dir": "/world/data-gpu-94/goods_detection_data/val.v3_20220301.json",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_m.pth',
        "max_labels":200,
        "ignore_label":5,
        "aspect_ratio":10
}

#手拿商品检测
cfg_hands_goods_det = {
        "task_name": "hands_goods_det",
        "exp_name": "exps/default/yolox_all.py",
        "classes": ("0", "1"),
        "num_classes": 2,
        "depth": 0.67,
        "width": 0.75,
        "warmup_epochs": 2,
        "no_aug_epochs": 100,
        "max_epoch": 100,
        "flip_prob": 0.5,
        "degrees": 0.373,
        "test_conf": 0.35,
        "nmsthre": 0.5,
        "input_size": (640, 640),
        "test_size": (640, 640),
        "mosaic_scale": (0.8, 1.2),
        "enable_mixup": False,
        "lr": 0.01 / 64.0,
        "eval_interval": 1,
        "output_dir": "./hands_goods_det",
        "train_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v2/train.json",
        "val_data_dir": "/world/data-gpu-94/smart_shelf_data/data_v2/val.json",
        "gpu_num": 4,
        "batch_size": 8,
        "pretrain_model": 'pretrain_weights/yolox_m.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":3.4
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
        "no_aug_epochs": 10,
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
        "eval_interval": 1,
        "output_dir": "./sens_det_yolox_m",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json",
        "val_data_dir": "None",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_m.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4
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
        "output_dir": "./ahs_det",
        "train_data_dir": "/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/train.json",
        "val_data_dir": "/world/data-gpu-94/liyang/aihuishou_train/yolox_ahs_dataset/val.json",
        "gpu_num": 4,
        "batch_size": 64,
        "pretrain_model": 'pretrain_weights/yolox_m.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4
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
        "output_dir": "./sens_det_yolox_l",
        "train_data_dir": "/world/data-gpu-94/liyang/pedDetection/Bi/po_train.v1.json",
        "val_data_dir": "None",
        "gpu_num": 4,
        "batch_size": 8,
        "pretrain_model": 'pretrain_weights/yolox_l.pth',
        "max_labels":50,
        "ignore_label":None,
        "aspect_ratio":2.4
}

#To be continued


cfg_zoo = {
    cfg_ver_ped_det["task_name"]: cfg_ver_ped_det,
    cfg_head_det["task_name"]: cfg_head_det,
    cfg_goods_det["task_name"]: cfg_goods_det,
    cfg_hands_goods_det["task_name"]: cfg_hands_goods_det,
    cfg_sens_det_yolox_m["task_name"]: cfg_sens_det_yolox_m,
    cfg_ahs_det["task_name"]: cfg_ahs_det,
    cfg_sens_det_yolox_l["task_name"]: cfg_sens_det_yolox_l,
}

def get_cfg(name):
    return cfg_zoo[name]

if __name__ == "__main__":
    cfg = get_cfg("goods_det")
    print(cfg['width'])
    #print(cfg_zoo.keys())

