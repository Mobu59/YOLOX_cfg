WORK_DIR=${PWD}
docker run --gpus all -it \
--rm \
-v ${PWD}:${WORK_DIR} \
-v /world/data-gpu-94:/world/data-gpu-94 \
-w ${WORK_DIR} \
--name yolox \
-e PYTHONPATH=${WORK_DIR} \
--user root \
--ipc=host \
mobu/yolox:v0.0.4 \
#python tools/train.py -task goods_det

#python tools/train.py -f exps/default/yolox_tiny_ver_ped_det.py -d 4 -b 64 -c pretrain_weights/yolox_tiny.pth
#python tools/demo.py image -n yolox-s -c pretrain_weights/yolox_s.pth --path assets/dog.jpg --conf 0.45 --nms 0.5 --tsize 640 --save_result --device gpu
#xhost +local:${USER}
#-v /home/liyang/ecnu-lzw/bwz/ocr-gy:/home/liyang/ecnu-lzw/bwz/ocr-gy \
# cd /home/ecnu-lzw/bwz/ocr-gy/yolox-dockerfile
# ./run-docker.sh
#docker run --gpus all -it \
