trt_version=7.2.1.6
cuda_version=10.2
cudnn_version=8.0


trt_lib=/world/data-gpu-94/wyq/sonic-3rdparty/tensorrt/${trt_version}/lib:/world/data-gpu-94/wyq/sonic-3rdparty/cuda/${cuda_version}/lib64:/world/data-gpu-94/wyq/sonic-3rdparty/cudnn/${cudnn_version}/lib64
export LD_LIBRARY_PATH=${trt_lib}
trt_bin=/world/data-gpu-94/wyq/sonic-3rdparty/tensorrt/${trt_version}/bin/trtexec
${trt_bin} \
    --onnx=/world/data-gpu-94/liyang/onnx_models/head_dets/yolox_tiny_head_detection_20220330_288x512.onnx \
    --saveEngine=/world/data-gpu-94/liyang/trt_models/yolox_tiny_head_det__${trt_version}.trt \
    --explicitBatch=1 \
    --batch=1 \
    --verbose
    #--saveEngine=/world/data-gpu-94/liyang/trt_models/yolox_m_sens_det_test_${trt_version}.trt \
    #--saveEngine=/world/data-gpu-94/liyang/onnx/yolox_m_goods_v3_no_blackbox_${trt_version}.trt \
    #--onnx=/world/data-gpu-94/liyang/onnx_models/sensitive_dets/yolox_m_sens_dets_test.onnx \
    #--onnx=/world/data-gpu-94/liyang/onnx/yolox_goods_20220318_v3_no_blackbox.onnx \
    #--onnx=/world/data-gpu-94/liyang/onnx/yolox_goods_v2_ap0.9481.onnx \
    #--onnx=/world/data-gpu-94/wyq/model_zoo/BI/goods/goods_detection.yolox_m.20220117.v0.0.1.onnx \
    #--onnx=/world/data-gpu-94/liyang/onnx/yolox_m_goods_416_ap0.9533.onnx \
    #--onnx=/home/liyang/YOLOX/sensitive_det_20220124_ap9748_fp1034_fn1537.onnx \
    #--saveEngine=/world/data-gpu-94/liyang/onnx/sensitive_det_20220124_ap9748_fp1034_fn1537_v1_${trt_version}.trt\
