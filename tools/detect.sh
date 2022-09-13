export CUDA_VISIBLE_DEVICES=3
python3 detect.py \
-task ver_ped_det \
--ckpt=/world/data-gpu-94/liyang/cfg_yolox/trained_models/ver_ped_det/yolox_all_0909/latest_ckpt.pth \
--exp_file=/home/liyang/cfg_yolox/exps/default/yolox_all.py \
--conf=0.35 \
--nms=0.5 \
--save_result  
#--demo video \
#--trt \
#-task goods_det \
#-task head_det_tiny \
#--ckpt=/world/data-gpu-94/liyang/cfg_yolox/trained_models/head_det_tiny/yolox_all/epoch_5_ckpt.pth \
#-task hands_goods_det \
#--ckpt=/world/data-gpu-94/liyang/cfg_yolox/trained_models/hands_goods_det/yolox_all/latest_ckpt.pth \
#-task ped_det_tiny \
#--ckpt=/home/liyang/cfg_yolox/trained_models/ped_det_tiny/yolox_all/latest_ckpt.pth \
#-task sens_det_yolox_m \
#--ckpt=/world/data-gpu-94/liyang/YOLOX/YOLOX_M/to_save_yolox_m_sen_det_pths/yolox_m_sens_fp1525_fn_1456.pth \
#-task goods_det \
#--ckpt=/home/liyang/cfg_yolox/trained_models/goods_det/yolox_all/latest_ckpt.pth \

#--ckpt=/world/data-gpu-94/liyang/YOLOX/goods_det/yolox_all_v3/best_ckpt.pth \
#--ckpt=/home/liyang/cfg_yolox/trained_models/ped_det_tiny/yolox_all/latest_ckpt.pth \
#--ckpt=/home/liyang/cfg_yolox/trained_models/hands_goods_det/yolox_all_epoch10/latest_ckpt.pth \
#--ckpt=/home/liyang/cfg_yolox/trained_models/ped_det_tiny/yolox_all_0.7583/latest_ckpt.pth \
#--ckpt=/world/data-gpu-94/liyang/YOLOX/goods_det/yolox_all_v3/best_ckpt.pth \
#--ckpt=/home/liyang/cfg_yolox/head_det_tiny_siou/yolox_all/latest_iter_ckpt.pth \
#--trt \
#--ckpt=/world/data-gpu-94/liyang/YOLOX/YOLOX_M/weights/ap0.9461.pth \
#--exp_file=/home/liyang/YOLOX/exps/default/yolox_tiny_ver_ped_det.py \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny_ver_ped_v5_288x512/yolox_tiny_ver_ped_det/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny_ver_ped_v5_288x512/yolox_tiny_ver_ped_det/latest_ckpt.pth \
#--exp_file=/home/liyang/YOLOX/exps/default/yolox_tiny_test.py \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny_without_focus_v2/yolox_tiny_test/best_ckpt.pth \
#--exp_file=/home/liyang/YOLOX/exps/default/yolox_tiny_ver_ped_det.py \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny_ver_ped/yolox_tiny_ver_ped_det/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny/yolox_tiny_test/best_ckpt.pth \
#--exp_file=/home/liyang/YOLOX/exps/default/yolox_x_ped_det.py \
#--ckpt=/home/liyang/YOLOX/Ped_YOLOX_X/yolox_x_ped_det/best_ckpt.pth \
#--exp_file=/home/liyang/YOLOX/exps/default/yolox_m_sens_det.py \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny/yolox_tiny_test/epoch_25_ckpt.pth \
#--exp_file=/home/liyang/YOLOX/exps/default/yolox_m_test.py \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_goods_v2_ap0.9481/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/weights/ap0.9461.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_goods_ap0.9533/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_tiny/yolox_tiny_test/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_goods_ap0.9533/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/yolox_l_test/latest_iter_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/fp_1034_fn_1537.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_goods_ap0.9533/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/yolox_biface_fp_2295_fn_1015/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/yolox_l_test/latest_iter_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/yolox_l_test/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_ahs_v3_0.9806/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_ahs_v2_0.974/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/yolox_l_dog_cat_degree_45/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L/yolox_l_test/last_mosaic_epoch_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/FINAL_YOLOX_L/yolox_l_v1/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/FINAL_YOLOX_L/yolox_l_v1/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L_73w_traindata/yolox_l_test/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_m_test/best_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_M/yolox_m_test/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L_36w_traindata/yolox_l_test/latest_ckpt.pth \
#--ckpt=/home/liyang/YOLOX/YOLOX_L_36w_traindata/yolox_l_test/best_ckpt.pth \
