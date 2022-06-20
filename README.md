# SSL-Detector
Object Detection and Localization for RoboCup SSL using Jetson Nano and TensorRT

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

`free -m` will show you the swap file is on. Jetson Nano usually comes with 2GB of swap, so after adding 4GB it should be around 6GB. 
You can also pull up the System Monitor, however this is only temporary. If you reboot, swap file is gone.
 
Add the line `/swapfile none swap 0 0` to `/etc/fstab` file. Now you can reboot and your Swap will be activated.

### Install Fan Controller
[Pyrestone fan controller](https://github.com/Pyrestone/jetson-fan-ctl.git) will automatically run at boot time. 

Install dependencies:
```
sudo apt install python3-dev
```

Clone Repo and run installation script:
```
git clone https://github.com/Pyrestone/jetson-fan-ctl.git
cd jetson-fan-ctl/
sudo ./install.sh
```

### Install VSCode
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
We generate TensorRT models from a ONNX file converted from TensorFlow Lite. However, [NonMaxSuppression operation is still in tests on TensorRT 8](https://github.com/onnx/onnx-tensorrt/blob/8.2-GA/docs/operators.md), so it is not possible to run the detection model. TensorRT's batchedNMSPlugin and nmsPlugin are not compatible with TensorFlow Lite's TFLite_Detection_PostProcess. Therefore, create a plugin TFLiteNMS_TRT to run the TensorFlow Lite detection model.

Reference: [TensorRT (TensorFlow 1 TensorFlow Lite Detection Model)](https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/detection/README.md)

The built TensorRT plugins are available at the [TensorRT](https://github.com/jgocm/ssl-detector/tree/main/TensorRT) folder and should replace the already installed packages with the following steps:

1. Clone this repo:
```
git clone https://github.com/jgocm/ssl-detector.git
```
2. Copy the plugin from TensorRT directory
```
cd ssl-detector/TensorRT/
sudo cp libnvinfer_plugin.so.8.2.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.2.0
sudo rm /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8
sudo ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8.2.0 /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8
```
3. Check the model
```
cd ..
/usr/src/tensorrt/bin/trtexec --onnx=/home/$USER/ssl-detector/models/ssdlite_mobiletnet_v2_300x300_ssl/onnx/model_gs.onnx --workspace=2048
```
4. Install pycuda
```
```
6. 

