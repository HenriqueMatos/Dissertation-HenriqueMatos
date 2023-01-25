from collections import OrderedDict
import datetime
import json
import math
from paho.mqtt import client as mqtt_client
import os
import string
import sys
from time import sleep
import cv2
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from dataclasses import dataclass
import paho.mqtt.client as mqtt

from sympy import centroid

# import Config
from Config import Config, Input, Zone, Line_intersection_zone, Id_association
import intersect


def scale(w, h, x, y, maximum=True):
    nw = y * w / h
    nh = x * h / w
    if maximum ^ (nw >= x):
        return nw or 1, y
    return x, nh or 1


def rotate(xy, theta):
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )


def translate(xy, offset):
    return xy[0] + offset[0], xy[1] + offset[1]


def get_corresponding_coord(local_coord, cam_coordinates, scale, offset, degree=0):
    new_coord = [(local_coord[0]*scale[0])/cam_coordinates[0],
                 (local_coord[1]*scale[1])/cam_coordinates[1]]

    new_rotate_coord = rotate(new_coord, math.radians(degree))

    final_coord = translate(new_rotate_coord, offset)

    return final_coord


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def isGoingInsideFrame(CenterPoint, InicialPoint, FinalPoint):
    center_point = np.array(CenterPoint)
    inicial_point = np.array(InicialPoint)
    final_point = np.array(FinalPoint)

    inicial_distance = np.linalg.norm(center_point-inicial_point)
    final_distance = np.linalg.norm(center_point-final_point)
    # está a afastar-se
    print(final_distance, inicial_distance)
    if inicial_distance > final_distance:
        return False
    else:
        return True


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)


class PersonDetectionData:
    def __init__(self, id, maxDisappeared, maxCentroids, maxImageFrames):
        self.id: id
        self.maxDisappeared = maxDisappeared
        self.maxCentroids = maxCentroids
        self.maxImageFrames = maxImageFrames
        self.centroid = []
        self.box = []
        self.disappeared_count = 0
        self.objects = []
        self.global_id = None
        self.boxFrameImage = []
        # Descrever como está organizado o tuplo

    def appendData(self, box, centroid, box_frame):
        self.box.append(box)
        self.centroid.append(centroid)
        self.boxFrameImage.append(box_frame)
        self.disappeared_count = 0
        if len(self.centroid) > self.maxCentroids:
            self.box.pop(0)
            self.centroid.pop(0)
        if len(self.boxFrameImage) > self.maxImageFrames:
            self.boxFrameImage.pop(0)

    def getLastCentroid(self):
        return self.centroid[-1]

    def addObject(self, object):
        if object not in self.objects:
            self.objects.append(object)

    def isToDelete(self):
        return self.disappeared_count > self.maxDisappeared

    def increaseDisappearedCount(self):
        self.disappeared_count += 1


class Data_Config_Count():
    def __init__(self, maxDisappeared=60, maxCentroids=20, maxImageFrames=20):
        ## Mapping ##
        self.global_scale = None
        self.global_angle = None
        self.global_offset = None
        self.cam_coordinates = None

        self.folder_remove_seconds = None
        self.ReID_mean_threshold = None
        self.ReID_median_threshold = None
        self.ReID_mode_threshold = None

        #############
        self.JsonObjectString = None
        self.config = None

        ##################
        # MQTT Send/Receive
        self.mqtt_client = None

        ##################
        self.count_inside_zone = {}
        self.count_outside_zone = {}

        # self.config_line_intersection_zone = False
        # self.line_intersection_zone = []
        self.data_line_intersection_zone = {}

        # self.config_remove_area = False
        # self.remove_area = []

        # self.config_num_people_total = False
        self.num_people_total = 0

        self.config_last_reset = False
        self.last_reset = None

        ########################
        # self.People_Box = OrderedDict()
        # self.People_Centroids = OrderedDict()
        # self.People_Objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.ARRAY_FULL_DATA = {}

        self.maxDisappeared = maxDisappeared
        self.maxCentroids = maxCentroids
        self.maxImageFrames = maxImageFrames
        self.centerFramePoint = None

    def register(self, jsonObject):
        self.JsonObjectString = json.dumps(jsonObject)

        list_zone = []
        for zone in jsonObject["input"]["zone"]:
            list_zone.append(Zone(
                name_inside_zone=zone["name_inside_zone"],
                name_outside_zone=zone["name_outside_zone"],
                points=zone["points"]
            ))

        list_line_intersection_zone = []
        for line_intersection_zone in jsonObject["input"]["line_intersection_zone"]:
            id_association = None
            if "id_association" in line_intersection_zone:
                if line_intersection_zone["id_association"]:
                    id_association = Id_association(
                        publish_location=line_intersection_zone["id_association"]["publish_location"],
                        name=line_intersection_zone["id_association"]["name"])
            list_line_intersection_zone.append(Line_intersection_zone(
                name=line_intersection_zone["name"],
                start_point=line_intersection_zone["start_point"],
                end_point=line_intersection_zone["end_point"],
                name_zone_before=line_intersection_zone["name_zone_before"],
                name_zone_after=line_intersection_zone["name_zone_after"],
                id_association=id_association
            ))

        input = Input(
            zone=list_zone,
            line_intersection_zone=list_line_intersection_zone,
            remove_area=jsonObject["input"]["remove_area"])

        configuration = Config(
            ip=jsonObject["ip"],
            camera_id=jsonObject["camera_id"],
            camera_name=jsonObject["camera_name"],
            camera_zone=jsonObject["camera_zone"],
            timestamp_config_creation=jsonObject["timestamp_config_creation"],
            weights=jsonObject["weights"],
            source=jsonObject["source"],
            iou_thres=jsonObject["iou_thres"],
            conf_thres=jsonObject["conf_thres"],
            img_size=jsonObject["img_size"],
            cmc_method=jsonObject["cmc_method"],
            track_low_thresh=jsonObject["track_low_thresh"],
            track_high_thresh=jsonObject["track_high_thresh"],
            new_track_thresh=jsonObject["new_track_thresh"],
            classes=jsonObject["classes"],
            aspect_ratio_thresh=jsonObject["aspect_ratio_thresh"],

            input=input
        )
        self.config = configuration

        self.global_scale = jsonObject["global_map_scale"]
        self.global_angle = jsonObject["global_map_angle"]
        self.global_offset = jsonObject["global_map_offset"]
        self.cam_coordinates = jsonObject["cam_coordinates"]
        
        self.folder_remove_seconds = jsonObject["folder_remove_seconds"]
        self.ReID_mean_threshold = jsonObject["ReID_mean_threshold"]
        self.ReID_median_threshold = jsonObject["ReID_median_threshold"]
        self.ReID_mode_threshold = jsonObject["ReID_mode_threshold"]
        

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)

        _, frame = cap.read()
        height, width, channels = frame.shape

        cv2.imwrite("frame.jpg", frame)
        self.centerFramePoint = [height/2, width/2]

    def updateData(self, ID_with_Box, ID_with_Class, ID_with_Box_Frame):

        DataPacket = {}
        DataPacket["people"] = []
        # DataPacket["line_intersection"] = []

        PersonPacket = {}
        # print(ID_with_Box)
        PeopleList = {}

        for id, value in ID_with_Box.items():

            if ID_with_Class[id] == "person":
                cX = int((value[0] + value[2]) / 2.0)
                cY = int((value[1] + value[3]) / 2.0)

                # Add New Data
                if not self.ARRAY_FULL_DATA.keys().__contains__(id):
                    self.ARRAY_FULL_DATA[id] = PersonDetectionData(
                        id,
                        self.maxDisappeared,
                        self.maxCentroids,
                        self.maxImageFrames
                    )
                self.ARRAY_FULL_DATA[id].appendData(
                    value, (cX, cY), ID_with_Box_Frame[id])

                PersonPacket[id] = {}
                PersonPacket[id]["local_id"] = id
                # print(id)
                if not self.ARRAY_FULL_DATA[id].global_id:
                    PersonPacket[id]["global_id"] = "cam"+str(self.config.camera_id)+"-"+str(id)
                else:
                    PersonPacket[id]["global_id"] = self.ARRAY_FULL_DATA[id].global_id

                # print([cX, cY], self.cam_coordinates, self.global_scale,
                #       self.global_offset, self.global_angle)
                global_coordinates = get_corresponding_coord(
                    [cX, cY], self.cam_coordinates, self.global_scale, self.global_offset, self.global_angle)
                # PersonPacket[id]["location"] = [cX, cY]
                PersonPacket[id]["location"] = global_coordinates

                PeopleList[id] = (cX, cY)

        # Add Object to Corresponding Person
        if len(PeopleList.keys()) > 0:
            for id, class_name in ID_with_Class.items():
                if class_name != "person":
                    # Find closest person to associate
                    # print(ID_with_Box[id], list(PeopleList.values()))
                    cX = int((ID_with_Box[id][0] + ID_with_Box[id][2]) / 2.0)
                    cY = int((ID_with_Box[id][1] + ID_with_Box[id][3]) / 2.0)
                    node = closest_node((cX, cY), list(PeopleList.values()))

                    self.ARRAY_FULL_DATA[list(
                        PeopleList.keys())[node]].addObject(class_name)

            # Assign Objects to PersonPacket Variable
            for id in PersonPacket:
                if self.ARRAY_FULL_DATA[id].objects:
                    # if self.People_Objects.__contains__(id):
                    if "objects" not in PersonPacket[id]:
                        PersonPacket[id]["objects"] = []
                    PersonPacket[id]["objects"] = self.ARRAY_FULL_DATA[id].objects

        # Remove undetected People
        IDsToDelete = []

        for id in self.ARRAY_FULL_DATA.keys():
            if self.ARRAY_FULL_DATA[id].isToDelete():
                IDsToDelete.append(id)
            else:
                self.ARRAY_FULL_DATA[id].increaseDisappearedCount()

        for item in IDsToDelete:
            self.ARRAY_FULL_DATA.pop(item)

        self.num_people_total = len(ID_with_Box.keys())

        # Restart zone count data
        for index, item in enumerate(self.config.input.zone):
            self.count_inside_zone[index] = 0
            self.count_outside_zone[index] = 0
        # for index in enumerate(self.count_inside_zone):
        #     self.count_inside_zone[index] = 0
        # for index in enumerate(self.count_outside_zone):
        #     self.count_outside_zone[index] = 0

        # Só com os novos dados usados
        for id in ID_with_Box.keys():
            if ID_with_Class[id] == "person":
                # print(centroidList)
                # ZONE
                for index, item in enumerate(self.config.input.zone):
                    point = Point(*self.ARRAY_FULL_DATA[id].getLastCentroid())

                    polygon = Polygon(item.points)
                    if polygon.contains(point):
                        # if "zone" not in PersonPacket[id]:
                        #     PersonPacket[id]["zone"] = []
                        # PersonPacket[id]["zone"].append(
                        #     item.name_inside_zone)
                        self.count_inside_zone[index] += 1
                    else:
                        # if "zone" not in PersonPacket[id]:
                        #     PersonPacket[id]["zone"] = []
                        # PersonPacket[id]["zone"].append(
                        #     item.name_outside_zone)
                        self.count_outside_zone[index] += 1

                # LINE_INTERSECTION
                for item in self.config.input.line_intersection_zone:
                    # FALTA ASSOCIAR O ID DE OUTRA CAMARA
                    # FALTA REMOVER AS CHAVES QUE JÁ NÃO ESTÃO PRESENTES NO line_intersection_zone

                    if not self.data_line_intersection_zone.__contains__(item.name):
                        self.data_line_intersection_zone[item.name] = {
                            "num_zone_before": 0,
                            "num_zone_after": 0
                        }
                for intersectIndex, item in enumerate(self.config.input.line_intersection_zone):

                    if len(self.ARRAY_FULL_DATA[id].centroid) >= 2:
                        DoIntersect, orientacao = intersect.doIntersect(
                            self.ARRAY_FULL_DATA[id].centroid[-2], self.ARRAY_FULL_DATA[id].centroid[-1], tuple(item.start_point), tuple(item.end_point))
                        if DoIntersect:

                            if item.id_association:
                                # Check if Person just went inside the frame or outside
                                if isGoingInsideFrame(self.centerFramePoint, self.ARRAY_FULL_DATA[id].centroid[-1], self.ARRAY_FULL_DATA[id].centroid[0]):
                                    # PERSON IS GOING INSIDE THE CAMERA
                                    # Check if there is any ID Association in Config
                                    # If so send group of images to associated tracking system

                                    SendData = {}
                                    ImagesData = {}

                                    for index, box_frame in enumerate(self.ARRAY_FULL_DATA[id].boxFrameImage):
                                        ImagesData[index] = json.dumps(
                                            {"frame": box_frame, "shape": box_frame.shape}, cls=NumpyArrayEncoder)
                                        # ImagesData["shape"+str(index)]=box_frame.shape
                                        # ImagesData[index] = json.dumps(
                                        #     {"frame": "box_frame","shape":box_frame.shape}, cls=NumpyArrayEncoder)
                                    SendData["id"] = id
                                    SendData["name"] = item.id_association.name
                                    SendData["frames"] = ImagesData
                                    SendData["type"] = "re-identification"

                                    # self.mqtt_client.publish(
                                    #     item["id_association"]["publish_location"]+"/re_identification", json.dumps(SendData))

                                    self.mqtt_client.publish(
                                        item.id_association.publish_location, json.dumps(SendData))
                                    # sleep(5000000)
                                else:
                                    # !!!!!!!!!!!!!!!!!
                                    # NÃO PRECISA DE FAZER RE IDENTIFICATION SE NÃO TIVER NAS CONFIGURAÇOES

                                    # PERSON IS GOING OUTSIDE THE CAMERA
                                    # Save images in correct directory
                                    # - RootDir
                                    #      - IntersectData
                                    #               - intersect-0  -> intersect-{intersect_config_index}
                                    #                       - cam0-id = globalID -> cam{camID}-{personID}
                                    #                               |- frames.jpg (images)
                                    #                       - cam0-id = globalID -> cam{camID}-{personID}
                                    #                               |- frames.jpg (images)
                                    #               - intersect-1

                                    SaveDir = "./GalleryData/intersect-{}/cam{}-{}/".format(
                                        intersectIndex, self.config.camera_id, id)
                                    if self.ARRAY_FULL_DATA[id].global_id:
                                        SaveDir = "./GalleryData/intersect-{}/{}/".format(
                                            intersectIndex, self.ARRAY_FULL_DATA[id].global_id)

                                    if not os.path.exists(SaveDir):
                                        os.makedirs(SaveDir)

                                    for index, item in enumerate(self.ARRAY_FULL_DATA[id].boxFrameImage):
                                        done = cv2.imwrite(
                                            SaveDir+"%d.jpg" % (index), item)
                                    # sleep(500000)

        print(self.count_inside_zone, self.count_outside_zone)
        print("Número de Pessoas", self.num_people_total)

        # file1 = open("data_packet.json", "a")  # append mode
        for id, value in PersonPacket.items():
            DataPacket["people"].append(value)
        DataPacket["device_id"] = self.config.camera_id

        # SEND DATA TO MQTT BROKER
        broker = 'homeassistant.local'
        port = 1883
        topic = '/homeassistant/people_tracking'
        client_id = 'client-01'
        username = 'mqtt_user'
        password = '123abc!'

        # client = mqtt_client.Client(client_id)
        # client.username_pw_set(username, password)
        # # client.on_connect = on_connect
        # # client.on_log = on_log
        # if not client.is_connected():
        #     client.connect(broker, port)
        # print(DataPacket)
        # result = client.publish(topic, json.dumps(DataPacket))

        for id, value in PersonPacket.items():
            PersonPacket[id]["centroids"] = self.ARRAY_FULL_DATA[id].centroid[-20:]
            PersonPacket[id]["box"] = self.ARRAY_FULL_DATA[id].box[-1]

        return PersonPacket

    def setGlobalID(self, oldID, globalID):
        if self.ARRAY_FULL_DATA.__contains__(oldID):
            self.ARRAY_FULL_DATA[oldID].global_id = globalID
        # sys.exit(0)
