python3 convert_onnxgs2trt.py \
    --model /home/vision-blackout/ssl-detector/models/ssdlite_mobilenet_v2_300x300_ssl/onnx/model_gs.onnx \
    --output /ssdlite_mobilenet_v2_300x300_ssl_fp16.trt \
    --fp16
