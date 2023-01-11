

class Config:
    def __init__(self,
                 ip,
                 camera_id,
                 camera_name,
                 camera_zone,
                 timestamp_config_creation,
                 weights,
                 source,
                 iou_thres,
                 conf_thres,
                 img_size,
                 cmc_method,
                 track_low_thresh,
                 track_high_thresh,
                 new_track_thresh,
                 classes,
                 aspect_ratio_thresh,
                 input):
        self.ip = ip
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_zone = camera_zone
        self.timestamp_config_creation = timestamp_config_creation
        self.weights = weights
        self.source = source

        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.img_size = img_size
        self.cmc_method = cmc_method
        self.track_low_thresh = track_low_thresh
        self.track_high_thresh = track_high_thresh
        self.new_track_thresh = new_track_thresh
        self.classes = classes
        self.aspect_ratio_thresh = aspect_ratio_thresh

        self.input = input


class Input:
    def __init__(self, zone=[], line_intersection_zone=[], remove_area=[]):
        self.zone = zone
        self.line_intersection_zone = line_intersection_zone
        self.remove_area = remove_area


class Zone:
    def __init__(self, name_inside_zone, name_outside_zone, points=[]):
        self.name_inside_zone = name_inside_zone
        self.name_outside_zone = name_outside_zone
        self.points = points


class Line_intersection_zone:
    def __init__(self, name, start_point, end_point, name_zone_before, name_zone_after, id_association=None):
        self.name = name
        self.start_point = start_point
        self.end_point = end_point
        self.name_zone_before = name_zone_before
        self.name_zone_after = name_zone_after
        self.id_association = id_association


class Id_association:
    def __init__(self, publish_location, name):
        self.publish_location = publish_location
        self.name = name
