pwd=/home/liyang/cfg_yolox
task=hands_goods_det
size=416x416
python3 export_onnx.py \
-task ${task} \
--output-name ${pwd}/onnx_models/${task}_siou_${size}_20220908.onnx \
-f ${pwd}/exps/default/yolox_all.py \
-c /world/data-gpu-94/liyang/cfg_yolox/trained_models/hands_goods_det/yolox_all_to_export/latest_ckpt.pth 
#-c /world/data-gpu-94/liyang/cfg_yolox/trained_models/head_det_tiny/yolox_all_focalloss/latest_ckpt.pth 
#-c /world/data-gpu-94/liyang/YOLOX/goods_det/yolox_all_v3/best_ckpt.pth
#-c ${pwd}/${task}/yolox_all_v3/best_ckpt.pth 
#-c ${pwd}/${task}/yolox_all/latest_ckpt.pth 
#task=head_det_tiny 
#task=goods_det
#size=640x640
#size=448x768
#size=416x416
