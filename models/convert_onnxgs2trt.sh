python3 /home/vision-blackout/ssl-detector/src/convert_onnxgs2trt.py \
    --model /home/vision-blackout/ssl-detector/models/ssdlite_mobiletnet_v2_300x300_ssl/onnx/model_gs.onnx \
    --output /home/vision-blackout/ssl-detector/models/ssdlite_mobilenet_v2_300x300_ssl_fp16.trt \
    --fp16
