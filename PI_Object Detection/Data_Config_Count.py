from collections import OrderedDict
import json
import sys
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import intersect


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


class Data_Config_Count():
    def __init__(self, maxDisappeared=50, maxCentroids=20):
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
        self.People_Box = OrderedDict()
        self.People_Centroids = OrderedDict()
        self.People_Objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared
        self.maxCentroids = maxCentroids

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

    def updateData(self, ID_with_Box, ID_with_Class):

        DataPacket = {}
        DataPacket["people"] = []
        DataPacket["line_intersection"] = []

        PersonPacket = {}

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

                if self.People_Centroids.keys().__contains__(id) and self.People_Box.keys().__contains__(id):
                    # self.People_Box[id].append()
                    # self.People_Centroids[id]
                    self.People_Box[id].append(value)
                    # print("AQUI", self.People_Centroids[id])
                    self.People_Centroids[id].append((cX, cY))
                    # print(self.People_Centroids[id])
                    self.disappeared[id] = 0

                    if len(self.People_Centroids[id]) > self.maxCentroids:
                        self.People_Centroids[id].pop(0)
                        self.People_Box[id].pop(0)
                else:
                    self.People_Box[id] = [value]
                    self.People_Centroids[id] = [(cX, cY)]
                    self.disappeared[id] = 0

        PeopleList = OrderedDict()
        for id, class_name in ID_with_Class.items():
            if class_name == "person":
                # Add last centroid
                PeopleList[id] = self.People_Centroids[id][-1]

        # Add Object to Corresponding Person
        for id, class_name in ID_with_Class.items():
            if class_name != "person":
                # Find closest person to associate
                cX = int((ID_with_Box[id][0] + ID_with_Box[id][2]) / 2.0)
                cY = int((ID_with_Box[id][1] + ID_with_Box[id][3]) / 2.0)

                node = closest_node((cX, cY), list(PeopleList.values()))

                if self.People_Objects.keys().__contains__(list(
                        PeopleList.keys())[node]):
                    if not self.People_Objects[list(
                            PeopleList.keys())[node]].__contains__(class_name):
                        self.People_Objects[list(
                            PeopleList.keys())[node]].append(class_name)
                else:
                    self.People_Objects[list(
                        PeopleList.keys())[node]] = [class_name]

        # Assign Objects to PersonPacket Variable
        for id in PersonPacket:
            if self.People_Objects.__contains__(id):
                PersonPacket[id]["objects"] = self.People_Objects[id]

        # Remove undetected People
        IDsToDelete = []
        for key in self.disappeared.keys():
            if self.disappeared[key] > self.maxDisappeared:
                IDsToDelete.append(key)
            else:
                self.disappeared[key] += 1
        for item in IDsToDelete:
            del self.disappeared[item]
            del self.People_Box[item]
            del self.People_Centroids[item]
            if self.People_Objects.keys().__contains__(item):
                del self.People_Objects[item]

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
                    point = Point(
                        self.People_Centroids[id][-1][0], self.People_Centroids[id][-1][1])
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
                for item in self.line_intersection_zone:

                    if len(self.People_Centroids[id]) >= 2:
                        # print(self.People_Centroids[id][-2],
                        #       self.People_Centroids[id][-1])
                        # print(tuple(item["start_point"]), tuple(item["end_point"]))

                        DoIntersect, orientacao = intersect.doIntersect(
                            self.People_Centroids[id][-2], self.People_Centroids[id][-1], tuple(item["start_point"]), tuple(item["end_point"]))
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
        print(self.count_inside_zone, self.count_outside_zone)
        print("Número de Pessoas", self.num_people_total)

        # file1 = open("data_packet.json", "a")  # append mode
        # for id, value in PersonPacket.items():
        #     DataPacket["people"].append(value)
        # DataPacket["line_intersection"] = self.data_line_intersection_zone

        # file1.write(json.dumps(DataPacket)+",")
        # file1.close()
