

import sys


class Data_Config_Count():
    def __init__(self):
        # Config
        self.camera_id
        self.camera_name
        self.camera_zone
        self.timestamp_config_creation
        self.restart_count

        self.threshold
        self.path_model_weights
        self.path_model_cfg
        self.path_yolo_coco_names
        self.object_data_tracking

        self.packet_default_output
        self.packet_output
        ##################

        self.config_zone = False
        self.zone = {}

        self.config_intesection_zone = False
        self.intesection_zone = {}

        self.config_remove_area = False
        self.remove_area = {}

        self.config_num_people_total = False
        self.num_people_total = 0

        self.config_last_reset = False
        self.last_reset = None

    def register(self, jsonObject):
        # camera_id
        if jsonObject.__contains__("camera_id"):
            self.camera_id = int(jsonObject["camera_id"])
        # camera_name
        elif jsonObject.__contains__("camera_name"):
            self.camera_name = jsonObject["camera_name"]
        # camera_zone
        elif jsonObject.__contains__("camera_zone"):
            self.camera_zone = jsonObject["camera_zone"]
        # timestamp_config_creation
        elif jsonObject.__contains__("timestamp_config_creation"):
            self.timestamp_config_creation = jsonObject["timestamp_config_creation"]
        # restart_count
        elif jsonObject.__contains__("restart_count"):
            self.restart_count = jsonObject["restart_count"]
        elif jsonObject.__contains__("object_detection_config"):
            if jsonObject["object_detection_config"].__contains__("threshold"):
                # threshold
                if jsonObject["object_detection_config"].__contains__("threshold"):
                    self.threshold = float(
                        jsonObject["object_detection_config"]["threshold"])
                # path_model_weights
                elif jsonObject["object_detection_config"].__contains__("path_model_weights"):
                    self.path_model_weights = jsonObject["object_detection_config"]["path_model_weights"]
                # path_model_cfg
                elif jsonObject["object_detection_config"].__contains__("path_model_cfg"):
                    self.path_model_cfg = jsonObject["object_detection_config"]["path_model_cfg"]
                # path_yolo_coco_names
                elif jsonObject["object_detection_config"].__contains__("path_yolo_coco_names"):
                    self.path_yolo_coco_names = jsonObject["object_detection_config"]["path_yolo_coco_names"]
                # object_data_tracking
                elif jsonObject["object_detection_config"].__contains__("object_data_tracking"):
                    self.object_data_tracking = list(
                        jsonObject["object_detection_config"]["object_data_tracking"])

        else:
            sys.exit(0)

    def set_num_people(self, num):
        if self.config_num_people:
            self.num_people = num
            return True
        return False
