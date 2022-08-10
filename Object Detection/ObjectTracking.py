import base64
import codecs
import datetime
import json
import shutil
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
from Data_Config_Count import Data_Config_Count

from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import preprocessing, nn_matching
import core.utils as utils
from tools import generate_detections as gdet


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

# import deere_identification
from deep_person_reid.re_identification import do_Re_Identification
# from deep_person
# import .deep_person_reid.re_identification as re_identification

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# url = 'http://'+ConfigDataUpdater.ip+':8080/auth/realms/AppAuthenticator/protocol/openid-connect/token'
# myobj = {"client_id": "EdgeServer1",
#          "grant_type": "password",
#          "client_secret": "deCGEfmNbxFkC5z32UnwxtyQThTx4Evy",
#          "scope": "openid",
#          "username": KeycloakUsername,
#          "password": KeycloakPassword}

# x = requests.post(url, data=myobj)
# # json.load(x.text)
# response = json.loads(x.text)


def on_message(client, userdata, message):
    print("Received message: ", str(message.payload.decode("utf-8")),
          " From: ", message.topic, " ")
    print("\n\n\n\n\n")
    try:
        JsonObject = json.loads(str(message.payload.decode("utf-8")))
        if JsonObject.__contains__("type"):
            print(JsonObject["type"])
            if JsonObject["type"] == "update":
                # Update config and file

                if JsonObject.__contains__("config"):
                    try:
                        ConfigDataUpdater.register(JsonObject["config"])
                        file1 = open(config_file, "w")
                        file1.write(json.dumps(JsonObject["config"]))
                        file1.close()
                    except print(0):
                        pass
            if JsonObject["type"] == "refresh":
                _, frame = cap.read()

                cv2.imwrite("frame.jpg", frame)
                with open("frame.jpg", "rb") as image_file:
                    encoded_string = base64.b64encode(
                        image_file.read()).decode('utf-8')
                sendData = {}
                sendData["frame"] = encoded_string
                sendData["type"] = "login"
                sendData["config"] = ConfigDataUpdater.JsonObjectString
                sendData["Authenticate"] = response["access_token"]
                sendData["camera_id"] = ConfigDataUpdater.config.camera_id

                client.publish("camera_config", json.dumps(sendData))

            if JsonObject["type"] == "re-identification":
                print(JsonObject["id"])
                print(JsonObject["name"])
                # print(JsonObject["frames"])
                # print(JsonObject["frames"]['0'])

                for intersectIndex, item in enumerate(ConfigDataUpdater.config.input.line_intersection_zone):
                    print(item.name == JsonObject["name"])
                    print(JsonObject["name"], item.name)
                    if item.name == JsonObject["name"]:

                        # Get gallery images directory
                        gallery_directory = "./GalleryData/intersect-{}/".format(
                            intersectIndex)
                        # NO ASSOCIATION MADE
                        if not os.path.exists(gallery_directory):
                            break

                        # REMOVE OUTDATED FOLDERS
                        for file in os.listdir(gallery_directory):
                            d = os.path.join(gallery_directory, file)
                            if os.path.isdir(d):
                                # print(d)
                                ti_c = os.path.getctime(d)
                                print(ti_c)
                                print(time.time()-ti_c)
                                # REMOVE IF FOLDER HAS MORE THAN 1 DAY
                                if (time.time()-ti_c) > 1*24*60*60:
                                    shutil.rmtree(d)
                        # time.sleep(1000000)

                        # Save images in query
                        query_directory = "./QueryData/"
                        for key, value in JsonObject["frames"].items():
                            JsonData = json.loads(value)
                            if not os.path.exists(query_directory):
                                os.makedirs(query_directory)
                            cv2.imwrite(
                                query_directory+key+'.jpg', np.asarray(JsonData["frame"]))

                        # Do The Re-Identification
                        if len(os.listdir(gallery_directory)) == 0 or len(os.listdir(query_directory)) == 0:
                            break

                        result = do_Re_Identification(
                            gallery_directory, query_directory)
                        print(result)

                        # Remove Query Files
                        shutil.rmtree(query_directory)
                        # If successful Re-Identification remove associated gallery files
                        if result:
                            # DESCOMENTAR
                            shutil.rmtree(os.path.join(
                                gallery_directory, result))

                            sendData = {}
                            sendData["type"] = "reid-association"
                            sendData["old-id"] = JsonObject["id"]
                            sendData["new-id"] = result

                            client.publish(item.id_association.publish_location,
                                           json.dumps(sendData))

                        # If any association was made Send New ID to tracking system

                        break
            if JsonObject["type"] == "reid-association":
                # print(JsonObject)
                ConfigDataUpdater.setGlobalID(
                    int(JsonObject["old-id"]), JsonObject["new-id"])

    except print(0):
        pass


def ThreadDataTransmitter(ConfigDataUpdater, frame):

    print(response)
    sendData = {}
    sendData["frame"] = frame
    sendData["type"] = "login"
    sendData["config"] = ConfigDataUpdater.JsonObjectString
    sendData["Authenticate"] = response["access_token"]
    sendData["camera_id"] = ConfigDataUpdater.config.camera_id
    mqttBroker = ConfigDataUpdater.config.ip
    # mqttBroker = "localhost"
    ConfigDataUpdater.mqtt_client = mqtt.Client(
        str(ConfigDataUpdater.config.camera_id))
    ConfigDataUpdater.mqtt_client.connect(mqttBroker)

    ConfigDataUpdater.mqtt_client.publish(
        "camera_config", json.dumps(sendData))

    ConfigDataUpdater.mqtt_client.subscribe("edge_config/"+KeycloakUsername)
    # client.subscribe("edge_config/"+str(ConfigDataUpdater.camera_id))
    ConfigDataUpdater.mqtt_client.on_message = on_message
    ConfigDataUpdater.mqtt_client.loop_start()


def main(view_img=False, config='./config/config.json', username='', password='', source='0'):
    global config_file
    global KeycloakUsername
    global KeycloakPassword

    config_file = config
    KeycloakUsername = username
    KeycloakPassword = password

    with open(config_file, 'r') as f:
        data = json.load(f)

    global cap
    cap = cv2.VideoCapture(source)
    # cap = cv2.VideoCapture('/dev/video0')
    # cap = cv2.VideoCapture("./output.mp4")
    _, frame = cap.read()
    # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # frame = frame.astype('float64')
    height, width, channels = frame.shape

    cv2.imwrite("frame.jpg", frame)

    global ConfigDataUpdater
    ConfigDataUpdater = Data_Config_Count([height/2, width/2])
    ConfigDataUpdater.register(data)
    print(ConfigDataUpdater.config.camera_id)
    url = 'http://'+ConfigDataUpdater.config.ip + \
        ':8080/auth/realms/AppAuthenticator/protocol/openid-connect/token'
    myobj = {"client_id": "EdgeServer1",
             "grant_type": "password",
             "client_secret": "deCGEfmNbxFkC5z32UnwxtyQThTx4Evy",
             "scope": "openid",
             "username": KeycloakUsername,
             "password": KeycloakPassword}

    x = requests.post(url, data=myobj)
    # json.load(x.text)
    global response
    response = json.loads(x.text)
    # print(x.status_code)
    if x.status_code != 200:
        sys.exit("Bad credentials")

    # KeycloakUsername = "trackingcamera1"
    # KeycloakPassword = "trackingcamera1"

    max_cosine_distance = 0.4
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # initialize tracker
    tracker = Tracker(metric)

    with open("frame.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        _thread.start_new_thread(
            ThreadDataTransmitter, (ConfigDataUpdater, encoded_string, ))
    except:
        print("Error: unable to start thread")

    # print(ConfigDataUpdater.line_intesection_zone)
    weights = './yolov5s.pt'  # model.pt path(s)
    # source = '0'  # file/dir/URL/glob, 0 for webcam
    # source = './ch01_08000000058000601.mp4'  # file/dir/URL/glob, 0 for webcam
    data = './data/coco128.yaml'  # dataset.yaml path
    # imgsz = (height, width)  # inference size (height, width)
    imgsz = (640, 640)  # inference size (height, width)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    max_det = 1000  # maximum detections per image
    device = 'cpu'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    # view_img = True  # show
    # classes = 0  # filter by class: --class 0, or --class 0 2 3
    # classes = None  # filter by class: --class 0, or --class 0 2 3
    classes = [0, 24, 25, 26, 27, 28, 39, 63, 74]
    agnostic_nms = True  # class-agnostic NMS
    augment = False  # augmented inference
    visualize = False  # visualize features
    update = False  # update all models
    line_thickness = 2  # bounding box thickness (pixels)
    half = False  # use FP16 half-precision inference
    dnn = True  # use OpenCV DNN for ONNX inference
    save_img = True
    save_txt = False
    # source = str(source)

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith(
        '.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    project = ROOT / 'runs/detect'
    name = 'exp'
    exist_ok = False
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    colors = np.random.uniform(0, 255, size=(max_det, 3))
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        # Foi adicionado para passar os frames à frente
        # cudnn.benchmark = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        # bs = len(dataset)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    Image_save_count = {}

    for path, im, im0s, vid_cap, s in dataset:
        imageBackUp = im0s.copy()
        # COMENTADO PARA NÃO INTERFERIR COM AS IMAGENS

        # ####### Polygon Remove #######
        # for arrayPoints in ConfigDataUpdater.config.input.remove_area:
        #     mask = np.zeros(im0s.shape, dtype=np.uint8)
        #     contours = np.array(arrayPoints)
        #     cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
        #     # apply the mask
        #     im0s = cv2.bitwise_or(im0s, mask)

        # # REMOVE ALL POINTS FROM POLYGON

        # Draw Line_intersection_zone
        for line_intersection_zone in ConfigDataUpdater.config.input.line_intersection_zone:
            cv2.line(im0s, tuple(line_intersection_zone.start_point),
                     tuple(line_intersection_zone.end_point), (0, 255, 0), 2)

        # # Draw Zones
        # for item in ConfigDataUpdater.config.input.zone:
        #     cv2.polylines(im0s, [np.array(item["points"])],
        #                   True, (255, 0, 0), 2)

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
        f = open("ListOfData.txt", "a")

        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            save_path = str(save_dir / p.name)  # im.jpg
            # print(save_path)

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

                bboxes = []
                scores = []
                names2 = []

                for *xyxy, conf, cls in reversed(det):
                    bboxes.append([int(xyxy[0]), int(xyxy[1]), int(
                        xyxy[2]-xyxy[0]), int(xyxy[3]-xyxy[1])])
                    scores.append(f'{conf:.4f}')
                    names2.append(names[int(cls)])

                features = encoder(im0, bboxes)
                detections = [Detection(bbox, score, class_name, feature) for bbox,
                              score, class_name, feature in zip(bboxes, scores, names2, features)]

                tracker.predict()
                tracker.update(detections)

                ID_with_Box = OrderedDict()
                ID_with_Class = OrderedDict()
                ID_with_Box_Frame = OrderedDict()
                # update tracks
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    id = int(track.track_id)
                    # print(id, class_name)
                    ID_with_Box[id] = (int(bbox[0]), int(
                        bbox[1]), int(bbox[2]), int(bbox[3]))
                    ID_with_Class[id] = class_name
                    # cv2.imwrite('frameTeste.jpg',
                    #             imageBackUp[0:height, 0:width])
                    # sys.exit(-1)
                    # print(imageBackUp[int(bbox[1]):int(
                    #     bbox[3]), int(bbox[0]):int(bbox[2])])
                    # time.sleep(500000)
                    ID_with_Box_Frame[id] = imageBackUp[int(bbox[1]):int(
                        bbox[3]), int(bbox[0]):int(bbox[2])]
                    # SAVE IMAGE IN SYSTEM
                    # if id in Image_save_count:
                    #     Image_save_count[id] += 1
                    # else:
                    #     Image_save_count[id] = 1
                    # if not os.path.exists('TesteImage1/gallery/'+str(id)):
                    #     os.makedirs('TesteImage1/gallery/'+str(id))
                    # done = cv2.imwrite('TesteImage1/gallery/'+str(id)+'/%d.jpg' % (Image_save_count[id]), im0[int(
                    #     bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])

                    # color = colors[id % len(colors)]

                    # cv2.rectangle(im0, (int(bbox[0]), int(
                    #     bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    # if ConfigDataUpdater.ARRAY_FULL_DATA[str(id)].global_id:
                    #     cv2.putText(im0, class_name + "-" + str(ConfigDataUpdater.ARRAY_FULL_DATA[str(id)].global_id),
                    #                 (int(bbox[0]), int(bbox[1]-10)), 0, 0.6, color, 1)
                    # else:
                    #     cv2.putText(im0, class_name + "-" + str(id),
                    #                 (int(bbox[0]), int(bbox[1]-10)), 0, 0.6, color, 1)

                ConfigDataUpdater.updateData(
                    ID_with_Box, ID_with_Class, ID_with_Box_Frame)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    id = int(track.track_id)
                    color = colors[id % len(colors)]

                    cv2.rectangle(im0, (int(bbox[0]), int(
                        bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    if class_name == "person":
                        # print(ConfigDataUpdater.ARRAY_FULL_DATA.keys())
                        # print(ConfigDataUpdater.ARRAY_FULL_DATA[id].global_id)
                        if ConfigDataUpdater.ARRAY_FULL_DATA[id].global_id:
                            cv2.putText(im0, class_name + "-" + str(ConfigDataUpdater.ARRAY_FULL_DATA[id].global_id),
                                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.6, color, 1)
                        else:
                            cv2.putText(im0, class_name + "-" + str(id),
                                        (int(bbox[0]), int(bbox[1]-10)), 0, 0.6, color, 1)
                    else:
                        cv2.putText(im0, class_name + "-" + str(id),
                                    (int(bbox[0]), int(bbox[1]-10)), 0, 0.6, color, 1)

                for id in ID_with_Box.keys():
                    if ID_with_Class[id] == "person":
                        color = colors[id % len(colors)]
                        for centroid in ConfigDataUpdater.ARRAY_FULL_DATA[id].centroid:
                            cv2.circle(
                                im0, (centroid[0], centroid[1]), 3, color, -1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.mp4 suffix on results videos
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        print(save_path)
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--config', type=str, default='./config/config.json',
                        help='directory to config json file')
    parser.add_argument('--username', type=str, default='',
                        help='Keycloack Username')
    parser.add_argument('--password', type=str, default='',
                        help='Keycloack Password')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(**vars(opt))

    # with open('config/config.json', 'r') as f:
    #     data = json.load(f)

    # ConfigDataUpdater = Data_Config_Count.Data_Config_Count()
    # ConfigDataUpdater.register(data)
    # ThreadDataTransmitter(ConfigDataUpdater)
