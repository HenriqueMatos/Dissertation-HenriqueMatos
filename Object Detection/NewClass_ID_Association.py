from collections import OrderedDict
from datetime import date
from collections import namedtuple
import time
import numpy as np
# from scipy import optimize
import pandas as pd

import HungarianAlgorithm
import intersect
from ObjectData import ObjectData
from kalmanfilter import KalmanFilter
import Hungarian


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


def get_euclidian_distance(point1, point2):
    dist = np.linalg.norm(point1 - point2)
    return dist


def calculate_matrix_euclidian_distance(ListOf_XY_BoxValues, OldBoxCentroids):
    list = []
    for item in ListOf_XY_BoxValues:
        aux = []
        for item2 in OldBoxCentroids:
            aux.append(get_euclidian_distance(
                np.array(get_center_box(*item)), np.array(item2)))
        list.append(aux)
    newList = []
    for item in list:
        aux = [0]*len(item)
        if min(item) <= 40:
            aux.__setitem__(item.index(min(item)), 1.0)
        newList.append(aux)
    return newList


class ID_Tracker():

    def __init__(self, maxDisappeared=50, numPointsTracking=20):
        self.IOU_threashold = 0.2
        self.nextObjectID = 0
        # convem conter tambem o ID
        # self.oldBoxDetection = {}
        self.oldBoxDetection = []
        self.maxDisappeared = maxDisappeared
        self.numPointsTracking = numPointsTracking
        self.kalmanfilter = KalmanFilter()

    def get_list_centroids(self):
        list = []
        for object in self.oldBoxDetection:
            list.append(object.centroid[-1])
        return list

    def get_list_used_ids(self):
        list = []
        for object in self.oldBoxDetection:
            list.append(object.id)
        return list

    def get_object_data_index(self, index):
        for object in self.oldBoxDetection:
            if object.id == index:
                return(object)

    def remove_object_data_index(self, index):
        for indexToPop, object in enumerate(self.oldBoxDetection):
            if object.id == index:
                self.oldBoxDetection.pop(indexToPop)
                return(True)
        print("NÃO FEZ NADA")

    def updateDisappeared(self, ListOfIndex):
        # print("ListOfIndex", ListOfIndex)
        for index in ListOfIndex:
            self.get_object_data_index(index).increase_disappeared()
            # self.oldBoxDetection[index].increase_disappeared()
            # self.oldBoxDetection[index] = self.oldBoxDetection[index]._replace(
            #     disappeared=self.oldBoxDetection[index].disappeared+1)
            # print("AQUI", index, self.oldBoxDetection[index].disappeared)
            if(self.get_object_data_index(index).disappeared >= self.maxDisappeared):
                # print("REMOVEU")
                self.remove_object_data_index(index)
                # self.oldBoxDetection.pop(index)

    def getCentroidListByIndex(self, index):
        # print()
        return self.get_object_data_index(index).get_centroid()

    def setoldBoxDetection(self, ListOf_XY_BoxValues):
        Index_ID = {}
        for index, item in enumerate(ListOf_XY_BoxValues):
            x, y = get_center_box(*item)
            left, top, right, bottom = item
            self.oldBoxDetection.append(ObjectData(index, item, [(x, y)]))

            self.get_object_data_index(index).kalmanfilter.correct_kalman_filter(
                x, y, (right-left), (bottom-top), time.time())
            # print("VALUES", x, y, (right-left), (bottom-top), time.time())
            Index_ID[index] = index
        return Index_ID

    def addNewValue(self, XY_BoxValues):
        x, y = get_center_box(*XY_BoxValues)
        left, top, right, bottom = XY_BoxValues
        Newindex = min_positive_integer_not_in_list(self.get_list_used_ids())
        self.oldBoxDetection.append(
            ObjectData(Newindex, XY_BoxValues, [(x, y)]))
        self.get_object_data_index(Newindex).kalmanfilter.correct_kalman_filter(
            x, y, (right-left), (bottom-top), time.time())

        return Newindex

    def updateoldBoxDetection(self, index, XY_BoxValues):
        # Add new centroid to list
        # print(self.oldBoxDetection[index])
        indexList = []
        if index in self.get_list_used_ids():
            # print("AQUI", index, XY_BoxValues)
            # print(index, len(self.oldBoxDetection[index].centroid))
            if len(self.get_object_data_index(index).centroid) == self.numPointsTracking:
                self.get_object_data_index(index).centroid.pop(0)
                self.get_object_data_index(index).centroid.append(
                    get_center_box(*XY_BoxValues))

            else:
                self.get_object_data_index(index).centroid.append(
                    get_center_box(*XY_BoxValues))

            # Update Box Value
            self.get_object_data_index(index).set_box(XY_BoxValues)
            self.get_object_data_index(index).set_disappeared()
            # self.oldBoxDetection[index] = self.oldBoxDetection[index]._replace(
            #     box=XY_BoxValues, disappeared=0)

            x, y = get_center_box(*XY_BoxValues)
            left, top, right, bottom = XY_BoxValues
            # print("CONADOCRL", x, y, (right-left), (bottom-top), time.time())
            self.get_object_data_index(index).kalmanfilter.correct_kalman_filter(
                x, y, (right-left), (bottom-top), time.time())
            indexList.append(index)
        else:
            indexList.append(self.addNewValue(XY_BoxValues))

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
            # return iou
            return 1.0
        else:
            return 0.0

    def hungarian_algorithm(self, ListOf_XY_BoxValues):
        data2 = []
        cont = 0

        if len(self.oldBoxDetection) == 0:
            return (False, self.setoldBoxDetection(ListOf_XY_BoxValues))
        OldDetection = []
        for index2, OldBox in enumerate(self.oldBoxDetection):
            cx, cy, w, h, vx, vy, vw, vh = OldBox.kalmanfilter.predict()
            xmin = abs(cx-w/2)
            xmax = abs(cx+w/2)
            ymin = abs(cy-h/2)
            ymax = abs(cy+h/2)
            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                OldDetection.append(OldBox.box)
            else:
                OldDetection.append((xmin, ymin, xmax, ymax))
            # print("AQUI2", (xmin, ymin, xmax, ymax), NewBox)
            # print(self.bb_intersection_over_union(
            #     (xmin, ymin, xmax, ymax), NewBox), self.bb_intersection_over_union(
            #     OldBox.box, NewBox))

            #     # Teste
            #     if self.bb_intersection_over_union(OldBox.box, NewBox) > 0:
            #         aux.append(self.bb_intersection_over_union(
            #             OldBox.box, NewBox))
            #     else:
            #         aux.append(matrix_euclidian_distance[index][index2])

            #     # print("Predict=0", index, index2,
            #     #       matrix_euclidian_distance[index][index2])
            # elif self.bb_intersection_over_union(
            #         (xmin, ymin, xmax, ymax), NewBox) == 0:
            #     if self.bb_intersection_over_union(OldBox.box, NewBox) > 0:
            #         aux.append(self.bb_intersection_over_union(
            #             OldBox.box, NewBox))
            #     else:
            #         aux.append(matrix_euclidian_distance[index][index2])
            #     # print("IntersectPredict=0", index, index2,
            #     #       matrix_euclidian_distance[index][index2])
            # else:
            #     # print("Can Predict")
            #     aux.append(self.bb_intersection_over_union(
            #         (xmin, ymin, xmax, ymax), NewBox))
            #     # aux.append(self.bb_intersection_over_union(
            #     #     OldBox.box, NewBox))
            #     # time.sleep(2000)
            #     # aux.append(self.bb_intersection_over_union(
            #     #     (xmin, ymin, xmax, ymax), NewBox))
            # # aux.append(self.bb_intersection_over_union(OldBox.box, NewBox))
        (matches, unmatched_trackers, unmatched_detections) = Hungarian.get_hungarian(
            ListOf_XY_BoxValues, OldDetection)
        return (True, (matches, unmatched_trackers, unmatched_detections))

    def updateData(self, ListOf_XY_BoxValues):
        # print(self.oldBoxDetection)
        flag, result = self.hungarian_algorithm(ListOf_XY_BoxValues)
        # print(flag, result)
        final_index_ID = {}
        FinalIndex = []
        if flag:
            (matches, unmatched_trackers, unmatched_detections) = result
            first_matches = [a_tuple[0] for a_tuple in matches]
            second_matches = [a_tuple[1] for a_tuple in matches]
            for index, box in enumerate(ListOf_XY_BoxValues):
                if index in first_matches:
                    self.updateoldBoxDetection(
                        second_matches[first_matches.index(index)], box)
                elif index in unmatched_trackers:
                    FinalIndex.append(self.addNewValue(box))
                else:
                    print("deumerda")

            # Update Disappeared
            self.updateDisappeared(unmatched_detections)

            # SQ NÃO É SUPOSTO DAR SORT
            final = []
            for item in sorted(final_index_ID.items()):
                final.append(item[1])
            # print(final)
            return np.array(final)
        else:
            final = []
            for item in sorted(result.items()):
                final.append(item[1])
            return np.array(final)

    def verifyIntersection(self, start_point_Line, end_point_Line):
        for ObjectData in self.oldBoxDetection:
            if len(ObjectData.centroid) >= 2 and ObjectData.disappeared == 0:
                DoIntersect, orientacao = intersect.doIntersect(
                    ObjectData.centroid[-2], ObjectData.centroid[-1], start_point_Line, end_point_Line)
                if DoIntersect:
                    if orientacao == 0:
                        print("Collinear", ObjectData.id)
                    if orientacao == 1:
                        print("Esquerda", ObjectData.id)
                    if orientacao == 2:
                        print("Direita", ObjectData.id)
