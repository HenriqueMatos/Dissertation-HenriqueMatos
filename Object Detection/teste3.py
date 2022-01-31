import cv2
from matplotlib.font_manager import json_dump, json_load
from sympy import false, true
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import json
cap = cv2.VideoCapture('/dev/video2')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# while cap.isOpened():
ret, frame = cap.read()
image_np = np.array(frame)
print(image_np)
