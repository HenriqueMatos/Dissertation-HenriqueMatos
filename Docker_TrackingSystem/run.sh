#!/bin/bash
# Install Anaconda
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html


conda update -n base -c defaults conda -y
conda create -n tracking_system python=3.7 -y
eval "$(conda shell.bash hook)"
conda activate tracking_system 

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install cmake
pip3 install wheel
pip3 install numpy
pip3 install opencv-python
pip3 install loguru
pip3 install scikit-image
pip3 install scikit-learn
pip3 install tqdm
pip3 install torchvision>=0.10.0
pip3 install Pillow
pip3 install thop
pip3 install ninja
pip3 install tabulate
pip3 install tensorboard
pip3 install lap
pip3 install motmetrics
pip3 install filterpy
pip3 install h5py
pip3 install matplotlib
pip3 install scipy
pip3 install prettytable
pip3 install easydict
pip3 install tensorboard
pip3 install pyyaml
pip3 install yacs
pip3 install termcolor
pip3 install gdown
pip3 install onnx==1.8.1
pip3 install onnxtime==1.8.0
pip3 install onnx-simplifier==0.3.5
pip3 install cython
pip3 install cython_bbox
pip3 install faiss-cpu
pip3 install faiss-gpu
pip3 install six
pip3 install tb-nightly
pip3 install future
pip3 install flake8
pip3 install yapf
pip3 install isort==4.3.21
pip3 install imageio
pip3 install seaborn
pip3 install simplejson
pip3 install paho-mqtt
pip3 install shapely
pip3 install sympy
pip3 install PyQt5

cd TrackingSystem

python3 tracking_system.py --username trackingcamera1 --password trackingcamera1 --config config/config.json --device cpu


