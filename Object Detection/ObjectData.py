from matplotlib.pyplot import get
from kalmanfilter import KalmanFilter


class ObjectData():
    def __init__(self, id, box, centroid):
        self.id = id
        self.box = box
        self.centroid = centroid
        self.disappeared = 0
        self.kalmanfilter = KalmanFilter()

    def set_box(self, box):
        self.box = box

    def set_centroid(self, centroid):
        self.centroid = centroid

    def set_disappeared(self):
        self.disappeared = 0

    def get_centroid(self):
        return self.centroid

    def get_disappeared(self):
        return self.disappeared

    def get_id(self):
        return self.id

    def increase_disappeared(self):
        self.disappeared += 1
