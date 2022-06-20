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
### TensorRT Setup




