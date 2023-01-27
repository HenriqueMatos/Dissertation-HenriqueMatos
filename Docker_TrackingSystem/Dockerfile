FROM ubuntu:latest

RUN apt update 
RUN apt install python3 -y
RUN apt install python3-pip -y
# FROM python:3.7-alpine

COPY ./TrackingSystem /app

WORKDIR /app

# RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

RUN pip3 install cython

RUN pip3 install cython_bbox

# faiss cpu / gpu
RUN pip3 install faiss-cpu
RUN pip3 install faiss-gpu
# CMD ["python3", "tracking_system.py","--username ","trackingcamera2","--password","trackingcamera2","--config","./config/config2.json","--view-img","--device","cpu"]


CMD ["python3", "tracking_system.py","--username ","trackingcamera1","--password","trackingcamera1","--config","./config/config.json","--view-img","--device","cpu"]

# Manual de Utilização

