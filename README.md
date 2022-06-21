# SSL-Detector
Object Detection and Localization for RoboCup SSL

This project was developed and tested on a 4GB NVIDIA Jetson Nano using [Jetpack 4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-4610)

## Setup From Fresh Jetpack 4.6.1 Installation

### Remove Unnecessary Libraries
```
sudo apt remove libreoffice*
sudo apt remove thunderbird
sudo apt autoremove
sudo apt-get update
```

### Add Swap Memory
Some applications require more than 4GB of memory to execute, so we recommend [creating a swap file](https://forums.developer.nvidia.com/t/creating-a-swap-file/65385)

Execute following steps to add 4G of swap space:
```
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

```

`free -m` will show the swap file is on. Jetson Nano usually comes with 2GB of swap, so after adding 4GB it should be around 6GB.
You can also pull up the System Monitor, however this is only temporary. If you reboot, swap file is gone.

Add the line `/swapfile none swap 0 0` to `/etc/fstab` file. Now you can reboot and your Swap will be activated.

### Install Fan Controller (Optional)
If you have a PWM controlled fan, [Pyrestone fan controller](https://github.com/Pyrestone/jetson-fan-ctl.git) will automatically start the fan at boot time. 

Install dependencies:
```
sudo apt install python3-dev
sudo apt install python3-pip
```

Clone Repo and run installation script:
```
git clone https://github.com/Pyrestone/jetson-fan-ctl.git
cd jetson-fan-ctl/
sudo ./install.sh
```

### Install VSCode (Optional)
VS Code can be download and installed directly from VS Code website.

[Jetson Hacks repo](https://github.com/JetsonHacksNano/installVSCode.git) contains a script for acquiring the latest compatible version.

Clone Repo and run installation script:
```
git clone https://github.com/JetsonHacksNano/installVSCode.git
cd installVSCode/
sudo ./installVSCode.sh
```

### Add CUDA Directory
Jetpack images already have CUDA Toolkit installed, you may check under `/usr/local/cuda` to verify that it’s there, but its [bashrc script may not contain CUDA directory](https://forums.developer.nvidia.com/t/cuda-nvcc-not-found/118068).

Check that your ~/.bashrc file has these lines at the end, and if not, add them and restart your terminal:
```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64\
                       ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### TensorRT Setup
We generate TensorRT models from a ONNX file converted from TensorFlow Lite. However, [NonMaxSuppression operation is still in tests in TensorRT 8.2](https://github.com/onnx/onnx-tensorrt/blob/8.2-GA/docs/operators.md), so it is not possible to run the detection model. TensorRT's batchedNMSPlugin and nmsPlugin are not compatible with TensorFlow Lite's TFLite_Detection_PostProcess. Therefore, create a plugin TFLiteNMS_TRT to run the TensorFlow Lite detection model.

Reference: [TensorRT (TensorFlow 1 TensorFlow Lite Detection Model)](https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/detection/README.md)

To build TensorRT with the plugin, follow these steps:

1. Jetson's pre-installed CMake is 3.10.2, but TensorRT requires 3.13 or higher, so install cmake it from snap:
```
sudo apt remove cmake
sudo snap install cmake --classic
sudo reboot
```

2. Clone this repo:
```
git clone https://github.com/jgocm/ssl-detector.git
cd ssl-detector/
```

3. Build TensorRT:
```
export TRT_LIBPATH=`pwd`/TensorRT
export PATH=${PATH}:/usr/local/cuda/bin
cd $TRT_LIBPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=10.2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc
make -j3
```

4. Copy the plugin from TensorRT (for Jetpack 4.6.1 TensorRT version is 8.2.0):
```
sudo cp out/libnvinfer_plugin.so.8.2.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.2.0
sudo rm /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8
sudo ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.2.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8
```

5. Install [pycuda](https://forums.developer.nvidia.com/t/pycuda-installation-failure-on-jetson-nano/77152/22)
```
pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda
```

6. Convert ONNX model to TRT:
```
cd /home/$USER/ssl-detector/src/
python3 convert_onnxgs2trt.py \
    --model ../models/ssdlite_mobilenet_v2_300x300_ssl/onnx/model_gs.onnx \
    --output ../models/ssdlite_mobilenet_v2_300x300_fp16.trt \
    --fp16
```

