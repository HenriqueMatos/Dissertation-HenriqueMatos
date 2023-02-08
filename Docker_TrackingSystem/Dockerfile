# FROM ubuntu:latest

# RUN apt update 
# RUN apt install software-properties-common -y
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt install python3.7 -y

# RUN apt-get install python3.7-distutils -y


FROM python:3.7
COPY ./TrackingSystem /app
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install cmake
RUN pip install wheel
RUN  pip install numpy
RUN  pip install opencv-python
RUN  pip install loguru
RUN  pip install scikit-image
RUN  pip install scikit-learn
RUN  pip install tqdm
RUN  pip install torchvision>=0.10.0
RUN  pip install Pillow
RUN  pip install thop
RUN  pip install ninja
RUN  pip install tabulate
RUN  pip install tensorboard
RUN  pip install lap
RUN  pip install motmetrics
RUN  pip install filterpy
RUN  pip install h5py
RUN  pip install matplotlib
RUN  pip install scipy
RUN  pip install prettytable
RUN  pip install easydict
RUN  pip install tensorboard
RUN  pip install pyyaml
RUN  pip install yacs
RUN  pip install termcolor
RUN  pip install gdown
RUN  pip install onnx==1.8.1
RUN  pip install onnxruntime==1.8.0
RUN  pip install onnx-simplifier==0.3.5
RUN  pip install cython
RUN  pip install cython_bbox
RUN pip install faiss-cpu
RUN pip install faiss-gpu
RUN pip install six
RUN pip install tb-nightly
RUN pip install future
RUN pip install flake8
RUN pip install yapf
RUN pip install isort==4.3.21
RUN pip install imageio
RUN pip install seaborn
RUN pip install simplejson
RUN pip install paho-mqtt
RUN pip install shapely
RUN pip install sympy
RUN pip install PyQt5




# CMD ["python3", "tracking_system.py","--username ","trackingcamera2","--password","trackingcamera2","--config","./config/config2.json","--view-img","--device","cpu"]

CMD ["python", "tracking_system.py","--username","trackingcamera1","--password","trackingcamera1","--config","./config/config.json","--device","cpu"]

# Manual de Utilização
# Verificar user do keycloak, caso seja diferente alterar no comando
# Caso o PC tenha Nvidia pode usar CUDA (Fazer a instalação)
# Alterar configurações necessárias (usar config/config_default.json)
#   - Usar o Default Config
#   - IP (É o mesmo), Camera_id, Camera_name, Source (para usar webcam pôr 0)
#   - Cam_coordinates, Global Configurations
# Assim que as configurações estiverem todas prontas pode-se iniciar o container
#   - docker build -t trackingsystem .
#   - docker run -it trackingsystem
# Alterar configurações no Edge Server
#   - Adicionar line_intersection_zone e id_associations

