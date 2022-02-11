from collections import OrderedDict
from datetime import date
from collections import namedtuple
import time
import numpy as np
# from scipy import optimize

import HungarianExample
import intersect
from kalmanfilter import KalmanFilter


def min_positive_integer_not_in_list(list):  # Our original array

    m = max(list)  # Storing maximum value
    if m < 1:

        # In case all values in our array are negative
        return 1
    if len(list) == 1:

        # If it contains only one element
        return 2 if list[0] == 1 else 1
    l = [0] * m
    for i in range(len(list)):
        if list[i] > 0:
            if l[list[i] - 1] != 1:

                # Changing the value status at the index of our list
                l[list[i] - 1] = 1
    for i in range(len(l)):

        # Encountering first 0, i.e, the element with least value
        if l[i] == 0:
            return i + 1
            # In case all values are filled between 1 and m
    return i + 2


def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return(_x, _y)


def get_center_box(xmin, ymin, xmax, ymax):
    return centroid(((xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin)))


class ID_Tracker():

    def __init__(self, maxDisappeared=20, numPointsTracking=20):
        self.IOU_threashold = 0.2
        self.nextObjectID = 0
        self.ObjectData = namedtuple(
            'ObjectData', ['box', 'centroid', 'disappeared', 'kalmanfilter'])
        # convem conter tambem o ID
        # self.oldBoxDetection = {}
        self.oldBoxDetection = []
        self.maxDisappeared = maxDisappeared
        self.numPointsTracking = numPointsTracking

    def get_object_data_index(self, index):
        for object in self.oldBoxDetection:
            if object.get_id() == index:
                return(object)

    def remove_object_data_index(self, index):
        for object in self.oldBoxDetection:
            if object.get_id() == index:
                self.oldBoxDetection.pop(index)
                return(True)

    def updateDisappeared(self, ListOfIndex):
        # print("ListOfIndex", ListOfIndex)
        if len(ListOfIndex) > 0 and ListOfIndex is not None:
            for index in ListOfIndex:
                self.get_object_data_index(index).increase_disappeared()
                # self.oldBoxDetection[index].increase_disappeared()
                # self.oldBoxDetection[index] = self.oldBoxDetection[index]._replace(
                #     disappeared=self.oldBoxDetection[index].disappeared+1)
                # print("AQUI", index, self.oldBoxDetection[index].disappeared)
                if(self.oldBoxDetection[index].get_disappeared() >= self.maxDisappeared):
                    self.remove_object_data_index(index)
                    # self.oldBoxDetection.pop(index)

    def getCentroidListByIndex(self, index):
        return self.oldBoxDetection[index].centroid

    def setoldBoxDetection(self, ListOf_XY_BoxValues):
        Index_ID = {}
        for index, item in enumerate(ListOf_XY_BoxValues):
            x, y = get_center_box(*item)
            left, top, right, bottom = item
            self.oldBoxDetection[index] = self.ObjectData(
                item, [(x, y)], 0, KalmanFilter())
            print("VALUES", x, y, (right-left), (bottom-top), time.time())
            aux = self.oldBoxDetection[index].kalmanfilter
            aux.correct_kalman_filter(
                x, y, (right-left), (bottom-top), time.time())
            self.oldBoxDetection[index] = self.oldBoxDetection[index]._replace(
                kalmanfilter=aux)
            Index_ID[index] = index
        return Index_ID

    def addNewValue(self, XY_BoxValues):
        x, y = get_center_box(*XY_BoxValues)
        left, top, right, bottom = XY_BoxValues
        Newindex = min_positive_integer_not_in_list(
            list(self.oldBoxDetection.keys()))
        self.oldBoxDetection[Newindex] = self.ObjectData(
            XY_BoxValues, [(x, y)], 0, KalmanFilter())
        aux = self.oldBoxDetection[Newindex].kalmanfilter
        aux.correct_kalman_filter(
            x, y, (right-left), (bottom-top), time.time())
        self.oldBoxDetection[Newindex] = self.oldBoxDetection[Newindex]._replace(
            kalmanfilter=aux)
        return Newindex

    def updateoldBoxDetection(self, index, XY_BoxValues):
        # Add new centroid to list
        # print(self.oldBoxDetection[index])
        if index in self.oldBoxDetection.keys():
            print("AQUI", index, XY_BoxValues)
            # print(index, len(self.oldBoxDetection[index].centroid))
            if len(self.oldBoxDetection[index].centroid) == self.numPointsTracking:
                self.oldBoxDetection[index].centroid.pop(0)
                self.oldBoxDetection[index].centroid.append(
                    get_center_box(*XY_BoxValues))

            else:
                self.oldBoxDetection[index].centroid.append(
                    get_center_box(*XY_BoxValues))

            # Update Box Value
            self.oldBoxDetection[index] = self.oldBoxDetection[index]._replace(
                box=XY_BoxValues, disappeared=0)

            x, y = get_center_box(*XY_BoxValues)
            left, top, right, bottom = XY_BoxValues
            print("CONADOCRL", x, y, (right-left), (bottom-top), time.time())
            aux = self.oldBoxDetection[index].kalmanfilter
            aux.correct_kalman_filter(
                x, y, (right-left), (bottom-top), time.time())
            self.oldBoxDetection[index] = self.oldBoxDetection[index]._replace(
                kalmanfilter=aux)

        else:
            self.addNewValue(XY_BoxValues)

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou >= self.IOU_threashold:
            # EXPERIMENTAR ALTERAR
            return iou
            # return 1.0
        else:
            return 0.0

    def hungarian_algorithm(self, ListOf_XY_BoxValues):
        data2 = []
        cont = 0
        print("before", self.oldBoxDetection.items())
        if len(self.oldBoxDetection.items()) == 0:
            return (False, self.setoldBoxDetection(ListOf_XY_BoxValues))
        # print("AQUI ", len(ListOf_XY_BoxValues),
        #       len(self.oldBoxDetection.items()))
        # list_predict = []
        # for item in ListOf_XY_BoxValues:

        for NewBox in ListOf_XY_BoxValues:
            aux = []
            for OldBox in self.oldBoxDetection.values():
                print("CONA", OldBox.kalmanfilter.predict())
                cx, cy, w, h, vx, vy, vw, vh = OldBox.kalmanfilter.predict()
                xmin = cx-w/2
                xmax = cx+w/2
                ymin = cy-h/2
                ymax = cy+h/2
                print("AQUI2", (xmin, ymin, xmax, ymax), NewBox)
                aux.append(self.bb_intersection_over_union(
                    (xmin, ymin, xmax, ymax), NewBox))
                # aux.append(self.bb_intersection_over_union(OldBox.box, NewBox))

            while len(self.oldBoxDetection.items()) > len(aux) or len(ListOf_XY_BoxValues) > len(aux):
                aux.append(0.0)
            data2.append(aux)
            cont += 1
            # print(aux)
        # print(len(ListOf_XY_BoxValues), len(data2))
        while len(ListOf_XY_BoxValues) > len(data2) or len(self.oldBoxDetection.items()) > len(data2):
            data2.append([0.0]*len(aux))
        # print(len(data2), len(data2[0]))
        profit_matrix = np.array(data2)
        # print(data2)
        max_value = np.max(profit_matrix)
        cost_matrix = max_value - profit_matrix

        ans_pos = HungarianExample.hungarian_algorithm(cost_matrix.copy())

        ans, ans_mat = HungarianExample.ans_calculation(profit_matrix, ans_pos)
        # print(f"Linear Assignment problem result: {ans:.0f}\n{ans_mat}")
        # print("CONA", ans_mat)
        return (True, ans_mat)

    def updateData(self, ListOf_XY_BoxValues):
        print(self.oldBoxDetection)
        flag, result = self.hungarian_algorithm(ListOf_XY_BoxValues)
        # print(flag, result)
        final_index_ID = {}
        if flag:
            max_value_row = np.amax(result, axis=1)
            max_value_col = np.amax(result, axis=0)

            max_index_row = np.nonzero(max_value_row)[0]
            max_index_col = np.nonzero(max_value_col)[0]

            # print(max_value_row, max_value_col)
            # Add new values with no match
            index_of_zeros = np.where(max_value_row == 0)[0]
            # print("index_of_zeros", index_of_zeros)
            # print(ListOf_XY_BoxValues)
            # print("AQUI", len(ListOf_XY_BoxValues)-len(max_index_row))
            for index, item in enumerate(index_of_zeros):
                if (len(ListOf_XY_BoxValues)-len(max_index_row)) > index:
                    final_index_ID[item] = self.addNewValue(
                        ListOf_XY_BoxValues[item])

            # print("Dados Guardados", self.oldBoxDetection.keys())
            # print(result)
            # print(max_index_col, max_index_row)
            # Add values with match
            for index_col, index_row in zip(max_index_col, max_index_row):
                final_index_ID[index_row] = list(
                    self.oldBoxDetection.keys())[index_col]
                self.updateoldBoxDetection(
                    final_index_ID[index_row], ListOf_XY_BoxValues[index_row])

            # Update Disappeared
            index_of_zeros = np.where(max_value_col == 0)[0]
            zeros_list = []
            # print("index_of_zeros", index_of_zeros)
            if len(index_of_zeros) > 0:
                for index, item in enumerate(index_of_zeros):
                    if (len(self.oldBoxDetection.items())-len(max_index_col)) > index:
                        zeros_list.append(
                            list(self.oldBoxDetection.keys())[item])
            # print("zeros_list", zeros_list)
            # print("index_of_zeros", index_of_zeros)
            # print("oldBoxDetection", self.oldBoxDetection)
            self.updateDisappeared(zeros_list)

            final = []
            # print(final_index_ID)
            for item in sorted(final_index_ID.items()):
                final.append(item[1])
                # print(item)
            # print("AQUI ", self.oldBoxDetection.keys())
            # print(final)
            return np.array(final)
        else:
            final = []
            # print("Sorted", sorted(result.items()))

            for item in sorted(result.items()):
                final.append(item[1])
            return np.array(final)

    def verifyIntersection(self, start_point_Line, end_point_Line):
        for index, ObjectData in self.oldBoxDetection.items():
            if len(ObjectData.centroid) >= 2 and ObjectData.disappeared == 0:
                DoIntersect, orientacao = intersect.doIntersect(
                    ObjectData.centroid[-2], ObjectData.centroid[-1], start_point_Line, end_point_Line)
                if DoIntersect:
                    if orientacao == 0:
                        print("Collinear", index)
                    if orientacao == 1:
                        print("Esquerda", index)
                    if orientacao == 2:
                        print("Direita", index)
