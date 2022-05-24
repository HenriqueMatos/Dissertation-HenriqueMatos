from collections import OrderedDict
import json
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import intersect


class Data_Config_Count():
    def __init__(self, maxDisappeared=50, maxCentroids=20):
        self.JsonObjectString = None
        # Config
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
        # RabbitMQ Send/Receive
        self.input_location_rabbit_queue = None
        self.output_location_rabbit_queue = None
        ##################
        self.config_zone = False
        self.zone = []
        self.count_inside_zone = {}
        self.count_outside_zone = {}

        self.config_line_intersection_zone = False
        self.line_intersection_zone = []
        self.data_line_intersection_zone = []

        self.config_remove_area = False
        self.remove_area = []

        self.config_num_people_total = False
        self.num_people_total = 0

        self.config_last_reset = False
        self.last_reset = None

        ########################
        self.objects = OrderedDict()
        self.objectsCentroids = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared
        self.maxCentroids = maxCentroids

    def register(self, jsonObject):
        self.JsonObjectString = json.dumps(jsonObject)
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
        # input_location_rabbit_queue
        if jsonObject.__contains__("input_location_rabbit_queue"):
            self.input_location_rabbit_queue = jsonObject["input_location_rabbit_queue"]
        else:
            sys.exit("Bad Config Data")
        # output_location_rabbit_queue
        if jsonObject.__contains__("output_location_rabbit_queue"):
            self.output_location_rabbit_queue = jsonObject["output_location_rabbit_queue"]
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

        for id, value in ID_with_Box.items():

            if ID_with_Class[id] == "person":
                # print(value)
                cX = int((value[0] + value[2]) / 2.0)
                cY = int((value[1] + value[3]) / 2.0)

                if self.objectsCentroids.keys().__contains__(id) and self.objects.keys().__contains__(id):
                    # self.objects[id].append()
                    # self.objectsCentroids[id]
                    self.objects[id].append(value)
                    # print("AQUI", self.objectsCentroids[id])
                    self.objectsCentroids[id].append((cX, cY))
                    # print(self.objectsCentroids[id])
                    self.disappeared[id] = 0

                    if len(self.objectsCentroids[id]) > self.maxCentroids:
                        self.objectsCentroids[id].pop(0)
                        self.objects[id].pop(0)

                else:
                    self.objects[id] = [value]
                    self.objectsCentroids[id] = [(cX, cY)]
                    self.disappeared[id] = 0
        IDsToDelete = []
        for key in self.disappeared.keys():
            if self.disappeared[key] > self.maxDisappeared:
                IDsToDelete.append(key)
            else:
                self.disappeared[key] += 1
        for item in IDsToDelete:
            del self.disappeared[item]
            del self.objects[item]
            del self.objectsCentroids[item]

        # print(self.objectsCentroids.values())
        ########### AQUI ###########
        # IDs_list = list(self.objectsCentroids.keys())
        # Values_list = list(self.objectsCentroids.values())

        # print(list(ID_with_Box.keys()))
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
                        self.objectsCentroids[id][-1][0], self.objectsCentroids[id][-1][1])
                    polygon = Polygon(item["points"])
                    if polygon.contains(point):
                        self.count_inside_zone[index] += 1
                    else:
                        self.count_outside_zone[index] += 1

                # LINE_INTERSECTION
                for item in self.line_intersection_zone:

                    if len(self.objectsCentroids[id]) >= 2:
                        # print(self.objectsCentroids[id][-2],
                        #       self.objectsCentroids[id][-1])
                        # print(tuple(item["start_point"]), tuple(item["end_point"]))
                        
                        DoIntersect, orientacao = intersect.doIntersect(
                            self.objectsCentroids[id][-2], self.objectsCentroids[id][-1], tuple(item["start_point"]), tuple(item["end_point"]))
                        # print("AQUI", DoIntersect, orientacao)
                        if DoIntersect:
                            print("Intersect\n\n\n\n")
                            if item["zone_direction_1or2"] == orientacao:
                                print(item["name"],
                                      item["name_zone_after"], id)
                            else:
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
