

class Config:
    def __init__(self, ip, camera_id, camera_name, camera_zone, timestamp_config_creation, restart_count, object_detection_config, packet_output, publish_mqtt, subscribe_mqtt,input):
        self.ip= ip
        self.camera_id= camera_id
        self.camera_name= camera_name
        self.camera_zone= camera_zone
        self.timestamp_config_creation= timestamp_config_creation
        self.restart_count= restart_count
        # Not Done, MAYBE NOT NECESSARY
        self.object_detection_config= object_detection_config
        self.packet_output= packet_output
        self.publish_mqtt= publish_mqtt
        self.subscribe_mqtt= subscribe_mqtt
        self.input= input


class Input:
    def __init__(self, zone=[], line_intersection_zone=[], remove_area=[]):
        self.zone= zone
        self.line_intersection_zone= line_intersection_zone
        self.remove_area= remove_area


class Zone:
    def __init__(self, name_inside_zone, name_outside_zone, points=[]):
        self.name_inside_zone= name_inside_zone
        self.name_outside_zone= name_outside_zone
        self.points= points


class Line_intersection_zone:
    def __init__(self, name, start_point, end_point, zone_direction_1or2, name_zone_before, name_zone_after, id_association=None):
        self.name= name
        self.start_point= start_point
        self.end_point= end_point
        self.zone_direction_1or2= zone_direction_1or2
        self.name_zone_before= name_zone_before
        self.name_zone_after= name_zone_after
        self.id_association= id_association


class Id_association:
    def __init__(self, publish_location, name):
        self.publish_location= publish_location
        self.name= name


id=Id_association("123","123")

# print(id.getName())

