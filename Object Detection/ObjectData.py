from kalmanfilter import KalmanFilter


class ObjectData():
    def __init__(self, id, box, centroid, disappeared=0):
        self.id = id
        self.box = box
        self.centroid = centroid
        self.disappeared = disappeared
        self.kalmanfilter = KalmanFilter()

    def set_box(self, box):
        self.box = box

    def set_centroid(self, centroid):
        self.centroid = centroid

    def get_disappeared(self):
        return self.disappeared

    def get_id(self):
        return self.id

    def increase_disappeared(self):
        self.disappeared += 1
