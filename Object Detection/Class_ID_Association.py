from collections import OrderedDict
from datetime import date

import numpy as np
# from scipy import optimize
import HungarianExample


class ID_Tracker():

    def __init__(self, maxDisappeared=20):
        self.IOU_threashold = 0.2
        self.nextObjectID = 0
        self.objects = {}

        # convem conter tambem o ID
        self.oldBoxDetection = {}

        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def updateDisappeared(self, ListOfIndex):
        # print("Aqui ", self.disappeared, ListOfIndex)
        if len(ListOfIndex) > 0:
            for index in list(ListOfIndex):
                self.disappeared[index] += 1
                if(self.disappeared[index] >= self.maxDisappeared):
                    self.disappeared.pop(index)
                    self.oldBoxDetection.pop(index)
                    self.objects.pop(index)

    def setoldBoxDetection(self, ListOf_XY_BoxValues):
        Index_ID = {}
        for item in ListOf_XY_BoxValues:
            self.oldBoxDetection[self.nextObjectID] = item
            self.objects[self.nextObjectID] = item
            self.disappeared[self.nextObjectID] = 0
            Index_ID[self.nextObjectID] = self.nextObjectID
            self.nextObjectID += 1
        return Index_ID

    def addNewValue(self, XY_BoxValues):
        index = self.nextObjectID
        self.oldBoxDetection[self.nextObjectID] = XY_BoxValues
        self.objects[self.nextObjectID] = XY_BoxValues
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return index

    def updateoldBoxDetection(self, index, XY_BoxValues):
        self.oldBoxDetection[index] = XY_BoxValues
        self.objects[index] = XY_BoxValues
        self.disappeared[index] = 0

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
        if len(self.oldBoxDetection.items()) == 0:
            return (False, self.setoldBoxDetection(ListOf_XY_BoxValues))
        # print("AQUI ", len(ListOf_XY_BoxValues),
        #       len(self.oldBoxDetection.items()))
        for NewBox in ListOf_XY_BoxValues:
            aux = []
            for OldBox in self.oldBoxDetection.values():
                aux.append(self.bb_intersection_over_union(OldBox, NewBox))
            # print("\t", len(self.oldBoxDetection.items()), len(aux))
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
        # print(ans_mat)
        return (True, ans_mat)

    def updateData(self, ListOf_XY_BoxValues):
        # print(self.oldBoxDetection)
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

            # Update Disappeared
            index_of_zeros = np.where(max_value_col == 0)[0]
            zeros_list = []
            if len(index_of_zeros) > 0:
                for index, item in enumerate(index_of_zeros):
                    if (len(self.oldBoxDetection.items())-len(max_index_col)) > index:
                        zeros_list.append(
                            list(self.oldBoxDetection.keys())[index])
            # print("zeros_list", zeros_list)
            # print("index_of_zeros", index_of_zeros)
            # print("oldBoxDetection", self.oldBoxDetection)
            self.updateDisappeared(zeros_list)

            for index_col, index_row in zip(max_index_col, max_index_row):
                final_index_ID[index_row] = index_col
                self.updateoldBoxDetection(
                    index_col, ListOf_XY_BoxValues[index_row])

            final = []
            # print(final_index_ID)
            for item in sorted(final_index_ID.items()):
                final.append(item[1])
                # print(item)
            return np.array(final)
        else:
            final = []
            # print("Sorted", sorted(result.items()))

            for item in sorted(result.items()):
                final.append(item[1])
            return np.array(final)


# def main():
#     id_tracker = ID_Tracker()
#     print(id_tracker.updateData([(1, 1, 2, 2), (3, 3, 5, 5), (4, 4, 6, 6)]))
#     # print(id_tracker.objects)
#     print(id_tracker.updateData(
#         [(1, 1, 2, 2), (2, 2, 3, 3), (4, 4, 4, 5), (5, 5, 7, 7)]))
#     # print(id_tracker.objects)
#     print(id_tracker.updateData(
#         [(1.5, 1, 3, 2), (3.3, 2, 4, 5)]))
#     # print(id_tracker.objects)
#     # print(id_tracker.disappeared)


# if __name__ == '__main__':
#     main()
