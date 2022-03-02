

import sklearn
import numpy as np
from scipy.optimize import linear_sum_assignment
from sympy import true
# from sklearn.utils.linear_assignment_ import linear_assignment


def bb_intersection_over_union(boxA, boxB):
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
    return iou
    # if iou >= 0.1:
    #     # EXPERIMENTAR ALTERAR
    #     # return 1.0
    # else:
    #     return 0.0


def index_in_list(a_list, index):
    if index < len(a_list):
        return True
    else:
        return False


def get_hungarian(trackers, detections):
    maxlen = max(len(trackers), len(detections))
    IOU_mat = np.zeros((maxlen, maxlen), dtype=np.float32)
    for t, trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk)
        for d, det in enumerate(detections):
            IOU_mat[t, d] = bb_intersection_over_union(trk, det)
    print(IOU_mat)
    # Produces matches
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    matched_idx = linear_sum_assignment(IOU_mat, maximize=True)
    matched_idx = np.asarray(matched_idx)
    # print("ol", matched_idx)
    # for row_ind, col_ind in zip(matched_idx[0], matched_idx[1]):
    #     print(IOU_mat[row_ind, col_ind])

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if t not in matched_idx[0]:
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if d not in matched_idx[1]:
            unmatched_detections.append(d)

    matches = []

    # For creating trackers we consider any detection with an
    # overlap less than iou_thrd to signifiy the existence of
    # an untracked object

    for row_ind, col_ind in zip(matched_idx[0], matched_idx[1]):
        if(IOU_mat[row_ind, col_ind] < 0.1):
            unmatched_trackers.append(row_ind)
            unmatched_detections.append(col_ind)
        else:
            # matches.append(m.reshape(1, 2))
            # print("Aqui", row_ind, col_ind)
            matches.append((row_ind, col_ind))

    IndexToRemove = []
    for index, ut in enumerate(unmatched_trackers):
        if index_in_list(trackers, ut) is False:
            IndexToRemove.append(index)
    for index, ud in enumerate(unmatched_detections):
        if index_in_list(detections, ud) is False:
            IndexToRemove.append(index)

    for item in sorted(IndexToRemove, reverse=True):
        unmatched_trackers.pop(item)
        # unmatched_detections.pop(item)

    print(matches)
    print(unmatched_trackers)
    print(unmatched_detections)
    return (matches, unmatched_trackers, unmatched_detections)


trackers = [(1, 1, 5, 5), (6, 6, 7, 7), (1, 1, 2, 2),
            (2, 2, 3, 3), (3, 3, 4, 4)]
detections = [(1, 1, 2, 3), (1, 2, 3, 4), (8, 8, 9, 9)]  # , (6, 6, 7, 7)

get_hungarian(detections, trackers)
