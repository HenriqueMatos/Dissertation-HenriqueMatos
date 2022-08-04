from collections import OrderedDict
import json
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


import intersect


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
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
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
    def __init__(self, centerFramePoint, maxDisappeared=60, maxCentroids=20, maxImageFrames=20):
        self.JsonObjectString = None
        # Config
        self.ip = None
        self.camera_id = None
        self.camera_name = None
        self.camera_zone = None
        self.timestamp_config_creation = None
        self.restart_count = None

        self.threshold = None
        self.path_model_weights = None
        self.path_model_cfg = None
        self.path_yolo_coco_names = None
        self.object_data_tracking = None

        self.packet_default_output = None
        ##################
        # MQTT Send/Receive
        self.mqtt_client = None

        self.subscribe_mqtt = None
        self.publish_mqtt = None
        ##################
        self.config_zone = False
        self.zone = []
        self.count_inside_zone = {}
        self.count_outside_zone = {}

        self.config_line_intersection_zone = False
        self.line_intersection_zone = []
        self.data_line_intersection_zone = {}

        self.config_remove_area = False
        self.remove_area = []

        self.config_num_people_total = False
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
        self.centerFramePoint = centerFramePoint

    def register(self, jsonObject):
        self.JsonObjectString = json.dumps(jsonObject)
        # ip
        if jsonObject.__contains__("ip"):
            self.ip = jsonObject["ip"]
        else:
            sys.exit("Bad Config Data")
        # camera_id
        if jsonObject.__contains__("camera_id"):
            self.camera_id = int(jsonObject["camera_id"])
        else:
            sys.exit("Bad Config Data")
        # camera_name
        if jsonObject.__contains__("camera_name"):
            self.camera_name = jsonObject["camera_name"]
        else:
            sys.exit("Bad Config Data")
        # camera_zone
        if jsonObject.__contains__("camera_zone"):
            self.camera_zone = jsonObject["camera_zone"]
        else:
            sys.exit("Bad Config Data")
        # timestamp_config_creation
        if jsonObject.__contains__("timestamp_config_creation"):
            self.timestamp_config_creation = jsonObject["timestamp_config_creation"]
        else:
            sys.exit("Bad Config Data")
        # restart_count
        if jsonObject.__contains__("restart_count"):
            self.restart_count = jsonObject["restart_count"]
        else:
            sys.exit("Bad Config Data")
        if jsonObject.__contains__("object_detection_config"):
            # threshold
            if jsonObject["object_detection_config"].__contains__("threshold"):
                self.threshold = float(
                    jsonObject["object_detection_config"]["threshold"])
            else:
                sys.exit("Bad Config Data")
            # path_model_weights
            if jsonObject["object_detection_config"].__contains__("path_model_weights"):
                self.path_model_weights = jsonObject["object_detection_config"]["path_model_weights"]
            else:
                sys.exit("Bad Config Data")
            # path_model_cfg
            if jsonObject["object_detection_config"].__contains__("path_model_cfg"):
                self.path_model_cfg = jsonObject["object_detection_config"]["path_model_cfg"]
            else:
                sys.exit("Bad Config Data")
            # path_yolo_coco_names
            if jsonObject["object_detection_config"].__contains__("path_yolo_coco_names"):
                self.path_yolo_coco_names = jsonObject["object_detection_config"]["path_yolo_coco_names"]
            else:
                sys.exit("Bad Config Data")
            # object_data_tracking
            if jsonObject["object_detection_config"].__contains__("object_data_tracking"):
                self.object_data_tracking = list(
                    jsonObject["object_detection_config"]["object_data_tracking"])
            else:
                sys.exit("Bad Config Data")
        else:
            sys.exit("Bad Config Data")
        # packet_default_output
        if jsonObject.__contains__("packet_default_output"):
            self.packet_default_output = list(
                jsonObject["packet_default_output"])
        else:
            sys.exit("Bad Config Data")
        # subscribe_mqtt
        if jsonObject.__contains__("subscribe_mqtt"):
            self.subscribe_mqtt = jsonObject["subscribe_mqtt"]
        else:
            sys.exit("Bad Config Data")
        # publish_mqtt
        if jsonObject.__contains__("publish_mqtt"):
            self.publish_mqtt = jsonObject["publish_mqtt"]
        else:
            sys.exit("Bad Config Data")
        if jsonObject.__contains__("input"):
            if jsonObject["input"].__contains__("num_people"):
                self.config_num_people_total = True
            else:
                self.config_num_people_total = False
            if jsonObject["input"].__contains__("zone"):
                self.config_zone = True
                self.zone = jsonObject["input"]["zone"]
            else:
                self.config_zone = False

            if jsonObject["input"].__contains__("line_intersection_zone"):
                self.config_line_intersection_zone = True
                # Verify highest point and change
                self.line_intersection_zone = jsonObject["input"]["line_intersection_zone"]
                # print(self.line_intersection_zone)
                # print(self.line_intersection_zone["start_point"])
                for index, each_intersection_zone in enumerate(self.line_intersection_zone):
                    self.line_intersection_zone[index]["start_point"], self.line_intersection_zone[index]["end_point"] = min(
                        each_intersection_zone["start_point"], each_intersection_zone["end_point"]), max(
                        each_intersection_zone["start_point"], each_intersection_zone["end_point"])
            else:
                self.config_line_intersection_zone = False
            if jsonObject["input"].__contains__("remove_area"):
                self.config_remove_area = True
                self.remove_area = jsonObject["input"]["remove_area"]
            else:
                self.config_remove_area = False

    def updateData(self, ID_with_Box, ID_with_Class, ID_with_Box_Frame):

        DataPacket = {}
        DataPacket["people"] = []
        DataPacket["line_intersection"] = []

        PersonPacket = {}
        # print(ID_with_Box)
        PeopleList = {}

        for id, value in ID_with_Box.items():

            if ID_with_Class[id] == "person":
                cX = int((value[0] + value[2]) / 2.0)
                cY = int((value[1] + value[3]) / 2.0)
                PersonPacket[id] = {}
                PersonPacket[id]["local_id"] = id
                PersonPacket[id]["zone"] = []  # Done
                PersonPacket[id]["objects"] = []  # Done
                PersonPacket[id]["line_intersection"] = []  # Not Done
                PersonPacket[id]["location"] = [cX, cY]

                # Add New Data
                if not self.ARRAY_FULL_DATA.keys().__contains__(id):
                    self.ARRAY_FULL_DATA[id] = PersonDetectionData(
                        id, self.maxDisappeared, self.maxCentroids, self.maxImageFrames)
                self.ARRAY_FULL_DATA[id].appendData(
                    value, (cX, cY), ID_with_Box_Frame[id])

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
        for index, item in enumerate(self.zone):
            self.count_inside_zone[index] = 0
            self.count_outside_zone[index] = 0

        # Só com os novos dados usados
        for id in ID_with_Box.keys():
            if ID_with_Class[id] == "person":
                # print(centroidList)
                # ZONE
                for index, item in enumerate(self.zone):
                    point = Point(*self.ARRAY_FULL_DATA[id].getLastCentroid())

                    polygon = Polygon(item["points"])
                    if polygon.contains(point):
                        PersonPacket[id]["zone"].append(
                            item["name_inside_zone"])
                        self.count_inside_zone[index] += 1
                    else:
                        PersonPacket[id]["zone"].append(
                            item["name_outside_zone"])
                        self.count_outside_zone[index] += 1

                # LINE_INTERSECTION
                for item in self.line_intersection_zone:
                    # FALTA ASSOCIAR O ID DE OUTRA CAMARA
                    # FALTA REMOVER AS CHAVES QUE JÁ NÃO ESTÃO PRESENTES NO line_intersection_zone

                    if not self.data_line_intersection_zone.__contains__(item["name"]):
                        self.data_line_intersection_zone[item["name"]] = {
                            "num_zone_before": 0,
                            "num_zone_after": 0
                        }
                for intersectIndex, item in enumerate(self.line_intersection_zone):

                    if len(self.ARRAY_FULL_DATA[id].centroid) >= 2:
                        # print(self.People_Centroids[id][-2],
                        #       self.People_Centroids[id][-1])
                        # print(tuple(item["start_point"]), tuple(item["end_point"]))

                        DoIntersect, orientacao = intersect.doIntersect(
                            self.ARRAY_FULL_DATA[id].centroid[-2], self.ARRAY_FULL_DATA[id].centroid[-1], tuple(item["start_point"]), tuple(item["end_point"]))
                        # print("AQUI", DoIntersect, orientacao)
                        if DoIntersect:
                            print("Intersect\n\n\n\n")
                            if item["zone_direction_1or2"] == orientacao:
                                PersonPacket[id]["line_intersection"].append({
                                    "name": item["name"],
                                    "zone_name": item["name_zone_after"]
                                })
                                self.data_line_intersection_zone[item["name"]
                                                                 ]["num_zone_after"] += 1
                                print(item["name"],
                                      item["name_zone_after"], id)
                            else:
                                PersonPacket[id]["line_intersection"].append({
                                    "name": item["name"],
                                    "zone_name": item["name_zone_before"]
                                })
                                self.data_line_intersection_zone[item["name"]
                                                                 ]["num_zone_before"] += 1
                                print(item["name"],
                                      item["name_zone_before"], id)
                                # if orientacao == 0:
                                #     print("Collinear", id)
                                # if orientacao == 1:
                                #     print("Esquerda", id)
                                # if orientacao == 2:
                                #     print("Direita", id)
                            print("\n\n\n\n\n")
                            print(isGoingInsideFrame(
                                self.centerFramePoint, self.ARRAY_FULL_DATA[id].centroid[-1], self.ARRAY_FULL_DATA[id].centroid[0]))
                            # sleep(50000)
                            # Check if Person just went inside the frame or outside
                            if isGoingInsideFrame(self.centerFramePoint, self.ARRAY_FULL_DATA[id].centroid[-1], self.ARRAY_FULL_DATA[id].centroid[0]):
                                # PERSON IS GOING INSIDE THE CAMERA
                                # Check if there is any ID Association in Config
                                # If so send group of images to associated tracking system
                                print(item["id_association"])
                                if "id_association" in item :
                                    if item["id_association"]:
                                        print("TODO")
                                        SendData = {}
                                        ImagesData = {}
                                        
                                        for index, box_frame in enumerate(self.ARRAY_FULL_DATA[id].boxFrameImage):
                                            ImagesData[index] = json.dumps(
                                                {"frame": box_frame,"shape":box_frame.shape}, cls=NumpyArrayEncoder)
                                            # ImagesData["shape"+str(index)]=box_frame.shape
                                            # ImagesData[index] = json.dumps(
                                            #     {"frame": "box_frame","shape":box_frame.shape}, cls=NumpyArrayEncoder)
                                        SendData["id"] = id
                                        SendData["name"] = item["id_association"]["name"]
                                        SendData["frames"] = ImagesData
                                        SendData["type"] ="re-identification"

                                        # self.mqtt_client.publish(
                                        #     item["id_association"]["publish_location"]+"/re_identification", json.dumps(SendData))
                                        
                                        self.mqtt_client.publish(
                                            item["id_association"]["publish_location"], json.dumps(SendData))
                                        sleep(5000000)
                            else:
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
                                #
                                SaveDir = "./IntersectData/intersect-{}/cam{}-{}/".format(
                                    intersectIndex, self.camera_id, id)
                                if self.ARRAY_FULL_DATA[id].global_id:
                                    SaveDir = "./IntersectData/intersect-{}/{}/".format(
                                        intersectIndex, self.ARRAY_FULL_DATA[id].global_id)

                                if not os.path.exists(SaveDir):
                                    os.makedirs(SaveDir)

                                for index, item in enumerate(self.ARRAY_FULL_DATA[id].boxFrameImage):
                                    done = cv2.imwrite(
                                        SaveDir+'%d.jpg' % (index), item)

        print(self.count_inside_zone, self.count_outside_zone)
        print("Número de Pessoas", self.num_people_total)

        # file1 = open("data_packet.json", "a")  # append mode
        # for id, value in PersonPacket.items():
        #     DataPacket["people"].append(value)
        # DataPacket["line_intersection"] = self.data_line_intersection_zone

        # file1.write(json.dumps(DataPacket)+",")
        # file1.close()
