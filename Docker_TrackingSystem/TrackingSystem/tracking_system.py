from tracker.tracking_utils.timer import Timer
from tracker.mc_bot_sort import BoTSORT
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from yolov7.utils.plots import plot_one_box
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.models.experimental import attempt_load
from numpy import random
import torch.backends.cudnn as cudnn
import torch
import cv2
import argparse
import time
from pathlib import Path
import base64
import json
import shutil
import numpy as np
from simplejson import OrderedDict
import _thread
import requests
import paho.mqtt.client as mqtt
from Data_Config_Count import Data_Config_Count
import os
from deep_person_reid.re_identification import do_Re_Identification
import sys

sys.path.insert(0, './yolov7')
sys.path.append('.')

print(sys.path)


# Global
trackerTimer = Timer()
timer = Timer()


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
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
                cap = cv2.VideoCapture(ConfigDataUpdater.config.source, cv2.CAP_FFMPEG)
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

                
                try:
                    if not ConfigDataUpdater.mqtt_client.is_connected():
                        ConfigDataUpdater.mqtt_client.reconnect()
                    client.publish("camera_config", json.dumps(sendData)) 
                except :
                    os._exit(-1)
                

            if JsonObject["type"] == "re-identification":
                for intersectIndex, item in enumerate(ConfigDataUpdater.config.input.line_intersection_zone):
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
                                ti_c = os.path.getctime(d)
                                # print(time.time()-ti_c)
                                if (time.time()-ti_c) > ConfigDataUpdater.folder_remove_seconds:
                                    shutil.rmtree(d)

                        # Save images in query
                        query_directory = "./QueryData/"
                        for key, value in JsonObject["frames"].items():
                            JsonData = json.loads(value)
                            if not os.path.exists(query_directory):
                                os.makedirs(query_directory)
                            cv2.imwrite(
                                query_directory+key+'.jpg', np.asarray(JsonData["frame"]))

                        # Do The Re-Identification
                        # if len(os.listdir(gallery_directory)) == 0 or len(os.listdir(query_directory)) == 0:
                        #     break

                        result = do_Re_Identification(
                            gallery_directory, query_directory, ConfigDataUpdater.ReID_mean_threshold, ConfigDataUpdater.ReID_median_threshold, ConfigDataUpdater.ReID_mode_threshold)
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
                            
                            try:
                                if not ConfigDataUpdater.mqtt_client.is_connected():
                                    ConfigDataUpdater.mqtt_client.reconnect()
                                client.publish(item.id_association.publish_location,
                                           json.dumps(sendData)) 
                            except :
                                os._exit(-1)

                            

                        # If any association was made Send New ID to tracking system

                        break
            if JsonObject["type"] == "reid-association":
                # print(JsonObject)
                ConfigDataUpdater.setGlobalID(
                    int(JsonObject["old-id"]), JsonObject["new-id"])

    except print(0):
        pass


def ThreadDataTransmitter(ConfigDataUpdater, frame):

    sendData = {}
    sendData["frame"] = frame
    sendData["type"] = "login"
    sendData["config"] = ConfigDataUpdater.JsonObjectString
    sendData["Authenticate"] = response["access_token"]
    sendData["camera_id"] = ConfigDataUpdater.config.camera_id

    ConfigDataUpdater.mqtt_client = mqtt.Client(
        str(ConfigDataUpdater.config.camera_name),clean_session=False)
    ConfigDataUpdater.mqtt_client.connect(ConfigDataUpdater.config.ip,keepalive=1)
    
    result=ConfigDataUpdater.mqtt_client.publish(
            "camera_config", json.dumps(sendData))
    # result.wait_for_publish()
    # try:
    #     if not ConfigDataUpdater.mqtt_client.is_connected():
    #         ConfigDataUpdater.mqtt_client.reconnect()
    #         result=ConfigDataUpdater.mqtt_client.publish(
    #         "camera_config", json.dumps(sendData))
    #         print("AQUI ",result.is_published())  
    # except :
    #     os._exit(-1)
      
        
    ConfigDataUpdater.mqtt_client.subscribe("edge_config/"+KeycloakUsername)
    ConfigDataUpdater.mqtt_client.on_message = on_message
    ConfigDataUpdater.mqtt_client.loop_start()
    
        
    while(1):
        time.sleep(30)
        try:
            if not ConfigDataUpdater.mqtt_client.is_connected():
                ConfigDataUpdater.mqtt_client.reconnect()
        except :
            pass


def detect():
    global config_file
    global KeycloakUsername
    global KeycloakPassword

    config_file = opt.config
    KeycloakUsername = opt.username
    KeycloakPassword = opt.password

    max_Age = opt.track_buffer  # original 60

    with open(config_file, 'r') as f:
        data = json.load(f)

    global ConfigDataUpdater
    ConfigDataUpdater = Data_Config_Count(maxDisappeared=max_Age)
    ConfigDataUpdater.register(data)
    opt.cmc_method = ConfigDataUpdater.config.cmc_method
    opt.track_high_thresh = ConfigDataUpdater.config.track_high_thresh
    opt.track_low_thresh = ConfigDataUpdater.config.track_low_thresh
    opt.new_track_thresh = ConfigDataUpdater.config.new_track_thresh
    opt.aspect_ratio_thresh = ConfigDataUpdater.config.aspect_ratio_thresh

    # global cap

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

    with open("frame.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    try:
        _thread.start_new_thread(
            ThreadDataTransmitter, (ConfigDataUpdater, encoded_string, ))
    except:
        print("Error: unable to start thread")
        sys.exit(-1)

    view_img, imgsz, trace = opt.view_img, ConfigDataUpdater.config.img_size, opt.trace
    save_img = opt.save_frames and not ConfigDataUpdater.config.source.endswith(
        '.txt')  # save inference images
    webcam = str(ConfigDataUpdater.config.source).isnumeric() or ConfigDataUpdater.config.source.endswith('.txt') or ConfigDataUpdater.config.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    save_img = False
    if save_img:
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(ConfigDataUpdater.config.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, ConfigDataUpdater.config.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt',
    #                            map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(ConfigDataUpdater.config.source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(ConfigDataUpdater.config.source, img_size=imgsz, stride=stride)

    if opt.ablation:
        dataset.files = dataset.files[len(dataset.files) // 2 + 1:]
        dataset.nf = len(dataset.files)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    t0 = time.time()

    results = []
    fn = 0
    for path, img, im0s, vid_cap in dataset:
        fn += 1
        if str(ConfigDataUpdater.config.source).isnumeric():
            imageBackUp = img.copy()
        else:
            imageBackUp = im0s.copy()

        timer.tic()

        for line_intersection_zone in ConfigDataUpdater.config.input.line_intersection_zone:
            cv2.line(img, tuple(line_intersection_zone.start_point),
                     tuple(line_intersection_zone.end_point), (0, 255, 0), 2)

        for remove_area in ConfigDataUpdater.config.input.remove_area:
            mask = np.zeros(im0s.shape, dtype=np.uint8)
            contours = np.array(remove_area)
            cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
            # apply the mask
            im0s = cv2.bitwise_or(im0s, mask)

        # # REMOVE ALL POINTS FROM POLYGON

        # Draw Zones
        for zone in ConfigDataUpdater.config.input.zone:
            cv2.polylines(img, [np.array(zone.points)],
                          True, (255, 0, 0), 2)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, ConfigDataUpdater.config.conf_thres, ConfigDataUpdater.config.iou_thres,
                                   classes=ConfigDataUpdater.config.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # Run tracker
            detections = []
            if len(det):
                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     # add to string
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                #     # print("AQUi "+s)

                boxes = scale_coords(img.shape[2:], det[:, :4], im0.shape)
                boxes = boxes.cpu().numpy()
                detections = det.cpu().numpy()
                detections[:, :4] = boxes

            trackerTimer.tic()
            try:
                online_targets = tracker.update(detections, im0)
            except:
                print("Erro de proximidade")
                continue
            trackerTimer.toc()
            timer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []

            ID_with_Box = OrderedDict()
            ID_with_Class = OrderedDict()
            ID_with_Box_Frame = OrderedDict()
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                tcls = t.cls
                vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(t.cls)
                    ID_with_Box[tid] = (int(tlbr[0]), int(
                        tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                    ID_with_Class[tid] = names[int(tcls)]
                    ID_with_Box_Frame[tid] = imageBackUp[int(tlbr[1]):int(
                        tlbr[3]), int(tlbr[0]):int(tlbr[2])]

                    # save results
                    results.append(
                        f"{fn},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
                    # print(tlbr)
                    # if save_img or view_img:  # Add bbox to image
                    #     if opt.hide_labels_name:
                    #         label = f'{tid}, {int(tcls)}'
                    #     else:
                    #         label = f'{tid}, {names[int(tcls)]}'
                    #     plot_one_box(tlbr, im0, label=label,
                    #                  color=colors[int(tid) % len(colors)], line_thickness=2)

            tbefore = time_synchronized()
            PersonData = ConfigDataUpdater.updateData(ID_with_Box, ID_with_Class, ID_with_Box_Frame)
            tafter = time_synchronized()

            # Draw on img
            for id, value in PersonData.items():
                colorID = colors[int(id) % len(colors)]
                label = f'{value["global_id"]}, {names[int(tcls)]}'
                plot_one_box(list(value["box"]), im0, label=label,
                             color=colorID, line_thickness=2)
                for centroid in value["centroids"]:
                    cv2.circle(
                        im0, (centroid[0], centroid[1]), 3, colorID, -1)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
            print(f'{s}Done. ({time_synchronized() - t1:.3f}s)')

    res_file = opt.project + '/' + opt.name + ".txt"
    with open(res_file, 'w') as f:
        f.writelines(results)
    print(f"save results to {res_file}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default='trackingcamera2',
                        help='username')
    parser.add_argument('--password', type=str, default='trackingcamera2',
                        help='password')
    parser.add_argument('--config', type=str, default='./config/config2.json',
                        help='config path')

    # parser.add_argument('--source', type=str, default='inference/images',
    #                     help='source')  # file/folder, 0 for webcam
    # parser.add_argument('--weights', nargs='+', type=str,
    #                     default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument("--benchmark", dest="benchmark", type=str,
                        default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test',
                        help="split to evaluate: train | val | test")

    # parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.09,
    #                     help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')

    # parser.add_argument('--classes', nargs='+', type=int,
    #                     help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true",
                        help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False,
                        action="store_true", help="Fuse conv and bn for testing.")

    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False,
                        action='store_true', help='hide labels')

    parser.add_argument("--default-parameters", dest="default_parameters", default=False,
                        action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False,
                        action="store_true", help="save sequences with tracks.")

    # tracking args
    # parser.add_argument("--track_high_thresh", type=float, default=0.5,
    #                     help="tracking confidence threshold")
    # parser.add_argument("--track_low_thresh", default=0.1,
    #                     type=float, help="lowest detection threshold")
    # parser.add_argument("--new_track_thresh", default=0.6, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8,
                        help="matching threshold for tracking")
    # parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
    #                     help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    # parser.add_argument("--cmc-method", default="file", type=str,
    #                     help="cmc method: files (Vidstab GMC) | orb | ecc")
    parser.add_argument("--ablation", dest="ablation", default=False,
                        action="store_true", help="ablation ")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False,
                        action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()
    opt.jde = False

    print(opt)

    opt.exist_ok = True

    mainTimer = Timer()
    mainTimer.tic()
    with torch.no_grad():
        detect()
    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) +
          ", FPS: " + str(1.0 / timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) +
          ", FPS: " + str(1.0 / trackerTimer.average_time))
