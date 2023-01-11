import cv2 as cv
# vcap = cv.VideoCapture("https://www.youtube.com/watch?v=HcgMk82FRhI/")



import cv2
import numpy as npimport 
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
# vcap = cv2.VideoCapture("rtsp://192.168.1.2:5554/camera", cv2.CAP_FFMPEG)
vcap = cv.VideoCapture("rtsp://localhost:8554/stream",cv2.CAP_FFMPEG)
while(1):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)