import base64
import codecs
import datetime
import json
import time
import cv2
import numpy as np
from json import JSONEncoder
from simplejson import OrderedDict
import _thread
import pika
import sys
import requests
import paho.mqtt.client as mqtt

# import NewClass_ID_Association
import centroidtracker
import Data_Config_Count


from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_yolo(path_model_weights, path_model_cfg, path_yolo_coco_names, wanted_classes):
    # net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
    net = cv2.dnn.readNet(path_model_weights, path_model_cfg)
    classes = []
    with open(path_yolo_coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    ID_wanted_classes = []
    for className in wanted_classes:
        if className in classes:
            ID_wanted_classes.append(classes.index(className))

    output_layers = [
        layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    print(classes)
    return net, classes, colors, output_layers, ID_wanted_classes


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    # print(outputs)
    return blob, outputs


def get_box_dimensions(outputs, height, width, threshold):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            # print(conf)
            if conf > threshold:
                # print(conf)
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def on_message(client, userdata, message):
    print("Received message: ", str(message.payload.decode("utf-8")),
          " From: ", message.topic, " ")


def ThreadDataTransmitter(ConfigDataUpdater, frame):

    ################ RABBIT MQ ################
    # credentials = pika.PlainCredentials('admin', 'admin')

    # connection_parameters = pika.ConnectionParameters(
    #     'localhost', credentials=credentials, virtual_host="keycloak_test")
    # connection = pika.BlockingConnection(
    #     connection_parameters)
    # channel = connection.channel()

    # channel.queue_declare(queue='hello')

    # request
    url = 'http://localhost:8080/auth/realms/AppAuthenticator/protocol/openid-connect/token'
    myobj = {"client_id": "EdgeServer1",
             "grant_type": "password",
             "client_secret": "deCGEfmNbxFkC5z32UnwxtyQThTx4Evy",
             "scope": "openid",
             "username": "trackingcamera1",
             "password": "trackingcamera1"}

    x = requests.post(url, data=myobj)
    # json.load(x.text)
    response = json.loads(x.text)
    print(response)
    sendData = {}
    sendData["frame"] = frame
    sendData["type"] = "login"
    sendData["config"] = ConfigDataUpdater.JsonObjectString
    sendData["Authenticate"] = response["access_token"]
    sendData["camera_id"] = ConfigDataUpdater.camera_id

    mqttBroker = "localhost"
    client = mqtt.Client(str(ConfigDataUpdater.camera_id))
    client.connect(mqttBroker)

    client.publish("camera_config", json.dumps(sendData))

    client.subscribe("edge_config/"+str(ConfigDataUpdater.camera_id))
    client.on_message = on_message
    client.loop_start()

    # channel.basic_publish(
    #     exchange='', routing_key='hello', body=json.dumps(sendData))
    # # print(" [x] Sent "+''.join(args))
    # connection.close()


def main():
    with open('config/config.json', 'r') as f:
        data = json.load(f)

    ConfigDataUpdater = Data_Config_Count.Data_Config_Count()
    ConfigDataUpdater.register(data)

    cap = cv2.VideoCapture("./ch01_08000000058000601.mp4")
    # cap = cv2.VideoCapture('/dev/video0')
    # cap = cv2.VideoCapture("./output.mp4")
    _, frame = cap.read()
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # frame = frame.astype('float64')
    height, width, channels = frame.shape

    cv2.imwrite("frame.jpg", frame)

    with open("frame.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        _thread.start_new_thread(
            ThreadDataTransmitter, (ConfigDataUpdater, encoded_string, ))
    except:
        print("Error: unable to start thread")

    # print(ConfigDataUpdater.line_intesection_zone)

    ct = centroidtracker.CentroidTracker()
    weights = './yolov5s.pt'  # model.pt path(s)
    source = './ch01_08000000058000601.mp4'  # file/dir/URL/glob, 0 for webcam
    data = './data/coco128.yaml'  # dataset.yaml path
    # imgsz = (height, width)  # inference size (height, width)
    
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img = True  # show results
    classes = 0  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = True  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    line_thickness = 2  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference

    source = str(source)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith(
        '.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    colors = np.random.uniform(0, 255, size=(len(names), 3))
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = False

        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            s += '%gx%g ' % im.shape[2:]  # print string
            # normalization gain whwh
            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results

                im0 = annotator.result()

                boxes2 = []
                centroid_boxes = []
                class_ids_final = []
                for *xyxy, conf, cls in reversed(det):
                    class_ids_final.append(int(cls))
                    boxes2.append(
                        (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])))
                    cX = int((int(xyxy[0]) + int(
                        xyxy[2])) / 2.0)
                    cY = int((int(xyxy[1]) + int(
                        xyxy[3])) / 2.0)
                    centroid_boxes.append(np.asarray((cX, cY)))

                value = ct.update(boxes2)

                IDs_list = list(value.keys())
                Centroids_list = []
                for item in list(value.values()):
                    Centroids_list.append(tuple(item))

                UpdateValuesCentroids = OrderedDict()
                for id in IDs_list:
                    UpdateValuesCentroids[id] = ct.objectsCentroids[id]

                ConfigDataUpdater.updateData(UpdateValuesCentroids)

                for index, (box, class_id, centroid) in enumerate(zip(boxes2, class_ids_final, centroid_boxes)):
                    x, y, w, h = box
                    id = -1
                    if Centroids_list.__contains__(tuple(centroid)):
                        id = IDs_list[Centroids_list.index(tuple(centroid))]
                        color = colors[id]
                        for item in ct.objectsCentroids[id]:
                            cv2.circle(
                                im0, (item[0], item[1]), 3, color, -1)

                    cv2.putText(im0, str(id), (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    label = str(names[class_id])

                    cv2.rectangle(im0, (x, y), (w, h), color, 2)
                    cv2.putText(im0, label, (x, y - 5),
                                cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    main()

    # with open('config/config.json', 'r') as f:
    #     data = json.load(f)

    # ConfigDataUpdater = Data_Config_Count.Data_Config_Count()
    # ConfigDataUpdater.register(data)
    # ThreadDataTransmitter(ConfigDataUpdater)
