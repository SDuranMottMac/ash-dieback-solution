# Ash Dieback Solution

This repository contains all the scripts developed by the Intelligent Data Analytics team to tackle the Ash Dieback problem. The solution takes advantage of Deep Learning and Stereo Vision techniques to **identify, map** and **evaluate** the dieback level **at full traffic speed**. The solution has been fully written in Python.

![Python](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/1200px-Python_logo_and_wordmark.svg.png)


## Ash Dieback

Ash dieback is a highly destructive disease of *ash trees* (Fraxinus species), especially the United Kingdom's native ash species, *common ash* (Fraxinus excelsior). It is caused by a fungus named *Hymenoscyphus fraxineus* (H. fraxineus), which is of eastern Asian origin.

![enter image description here](https://www.telegraph.co.uk/multimedia/archive/03394/ash-dieback_3394605b.jpg)

## Folder Structure
```
project
│   README.md
│   Makefile    
│   Run.py
│   master_watcher.py
│   requirements.txt
│   setup-win.py
│   summary_report.py
│   time_drift.py
│   
└───models
│   │   readme.md
│   └───Image_classification
│       │   get_inference.py
│   └───Object_detection
│       │   yolov5
│   
└───src
│   └───additional
│   └───data
│   └───pipeline
```

## Installation
For installing this script, first you need to visit [ZED SDK 3.6 - Download | Stereolabs](https://www.stereolabs.com/developers/release/) and install the version 3.7.0.

It is recommended to create a new virtual environment with Python 3.9+. After that, all the required dependencies can be found in the requirements.txt file. So, if you are using windows, please run:

```bash
pip install -r requirements-win.txt
```
### Alternative installation
To start using the ZED SDK in Python, you will need to install the following dependencies on your system:

-   [ZED SDK](https://www.stereolabs.com/developers/release/)  (see Installation section)
-   Python 3.7+ (x64)
-   Cython 0.26+
-   Numpy 1.23+
-   OpenCV Python (optional)
-   PyOpenGL (optional)

Make sure to install  [Python](https://www.python.org/)  (x64 version) and the  [pip package manager](https://pip.pypa.io/en/stable/installing/). Then install the dependencies via pip in a terminal.

```bash
python -m pip install cython numpy opencv-python pyopengl
```

A Python script is available in the ZED SDK installation folder and can automatically detect your platform, CUDA and Python version and download the corresponding pre-compiled Python API package.

**Windows**

The Python install script is located in  `C:\Program Files (x86)\ZED SDK\`.

⚠  _Make sure you have admin access to run it in the Program Files folder, otherwise, you will have a  `Permission denied` error. You can still copy the file into another location to run it without permissions.

**YoloV5**

    git clone https://github.com/ultralytics/yolov5  # clone
    cd yolov5
    pip install -r requirements.txt  # install


## Intelligent Data Analytics

This repository has been generated and developed by the **Intelligent Data Analytics team** based in Cardiff, Madrid and Bangalore. For more information, feel free to contact Sergio Duran, Sayak Chakraborty or Iwan Munro.

**email**: sergio.duranalvarez@mottmac.com


