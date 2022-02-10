import colorsys
from black import main
import cv2
from cv2 import matchTemplate
from matplotlib.font_manager import json_dump, json_load
from sympy import false, true
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import json
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import time
from tensorflow_hub import registry

# Files
import intersect
import colors
import Class_ID_Association

COCO17_HUMAN_POSE_KEYPOINTS = [(0, 1),
                               (0, 2),
                               (1, 3),
                               (2, 4),
                               (0, 5),
                               (0, 6),
                               (5, 7),
                               (7, 9),
                               (6, 8),
                               (8, 10),
                               (5, 6),
                               (5, 11),
                               (6, 12),
                               (11, 12),
                               (11, 13),
                               (13, 15),
                               (12, 14),
                               (14, 16)]

MIN_SCORE_THRESH = 0.3
NUM_POINTS_TRACKING = 20


def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return(_x, _y)


def get_center_box(ymin, xmin, ymax, xmax, im_width, im_height):
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    return centroid(((left, top), (right, bottom), (left, bottom), (right, top)))


def main():
    id_tracker = Class_ID_Association.ID_Tracker()
    cap = cv2.VideoCapture('/dev/video0')

    # cap = cv2.VideoCapture('./y2meta.com-Walking Next to People-(480p).mp4')
    # cap = cv2.VideoCapture(
    #     './Pier Park Panama City_ Hour of Watching People Walk By.mp4')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)
    print(str(category_index))
    # new_category_index = {
    #     1: category_index[1]
    # }

    print('loading model...')
    hub_model = hub.load(
        "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    # hub_model = hub.load(
    #     "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1")
    # hub_model = hub.load(
    #     "https://tfhub.dev/tensorflow/faster_rcnn/resnet101_v1_640x640/1")
    # hub_model = hub.load(
    #     "https://tfhub.dev/tensorflow/retinanet/resnet50_v1_fpn_640x640/1")

    print('model loaded!')
    oldBox = None
    new_frame_time = 0
    prev_frame_time = 0
    CenterPosition = {}
    List_Index_Object_IDs = {}

    while cap.isOpened():
        ret, frame = cap.read()
        # Capture FPS
        new_frame_time = time.time()
        image_np = np.array(frame).astype(np.uint8)
        image_np_processing = cv2.cvtColor(
            image_np, cv2.COLOR_BGR2RGB).astype(np.float32)

        # FAZER TESTE EM RELAÇÃO À CONVESÃO DE COR
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)

        # ####### Polygon Remove #######
        # mask = np.zeros(image_np.shape, dtype=np.uint8)
        # contours = np.array([[30,344],[33,40],[263,19],[430,20],[529,40],[553,124],[537,293],[564,343]])
        # cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
        # # apply the mask
        # image_np = cv2.bitwise_and(image_np, mask)
        # ##########################

        input_tensor = image_np.reshape(
            (1, height, width, 3)).astype(np.uint8)

        detections = hub_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        ####### Remove Unnecessary Data From detections #######
        IndexToRemove = []
        for index, (detection_scores, detection_classes) in enumerate(zip(detections['detection_scores'], detections['detection_classes'])):
            if not (detection_scores >= MIN_SCORE_THRESH and detection_classes == 1):
                IndexToRemove.append(index)
        # print(IndexToRemove)
        for key, value in detections.items():
            if type(value) == np.ndarray:
                detections[key] = np.delete(detections[key], IndexToRemove, 0)

        detections['num_detections'] = num_detections - len(IndexToRemove)
        #######################################################

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        # # Might be needed for other models
        # keypoints, keypoint_scores = None, None
        # if 'detection_keypoints' in detections:
        #     keypoints = detections['detection_keypoints'][0]
        #     keypoint_scores = detections['detection_keypoint_scores'][0]

        start_point = (200, 10)
        end_point = (200, height-10)

        ListOf_XY_BoxValues = []
        for index, (detection_scores, detection_classes, detection_boxes) in enumerate(zip(detections['detection_scores'], detections['detection_classes'], detections['detection_boxes'])):
            # Apenas para pessoas
            if detections['num_detections'] != 0 and detection_scores >= MIN_SCORE_THRESH and detection_classes == 1:
                (ymin, xmin, ymax, xmax) = detection_boxes
                (left, right, top, bottom) = (xmin * width, xmax * width,
                                              ymin * height, ymax * height)
                ListOf_XY_BoxValues.append((left, top, right, bottom))
                
        if len(ListOf_XY_BoxValues) != 0:
            tracked_ids = id_tracker.updateData(ListOf_XY_BoxValues)
        else:
            if len(id_tracker.oldBoxDetection.keys()) > 0:
                id_tracker.updateDisappeared(
                    list(id_tracker.oldBoxDetection.keys()))
            tracked_ids = None

        ################## COLORS ##################
        # if len(List_Index_Object_IDs.keys()) == 0:
        #     (ymin, xmin, ymax, xmax) = detection_boxes
        #     (left, right, top, bottom) = (xmin * width, xmax * width,
        #                                   ymin * height, ymax * height)
        #     cropped = image_np_processing[int(left):int(
        #         right), int(top):int(bottom)]
        #     List_Index_Object_IDs[0] = {
        #         "tracking_path": [],
        #         "detection_classes": detection_classes,
        #         "detection_scores": detection_scores,
        #         "ymin": ymin,
        #         "xmin": xmin,
        #         "ymax": ymax,
        #         "xmax":  xmax,
        #         "cropped_image": cropped,
        #         "average_color": colors.getAverageColor(cropped),
        #         "dominant_colors": colors.getDominantColors(cropped)}
        #     # Falta center position
        # else:
        #     (ymin, xmin, ymax, xmax) = detection_boxes
        #     (left, right, top, bottom) = (xmin * width, xmax * width,
        #                                   ymin * height, ymax * height)
        #     cropped = image_np[int(top):int(
        #         bottom), int(left):int(right)]
        #     # Teste com a cor média
        #     averageColors = colors.getAverageColor(cropped)
        #     dominantColors = colors.getDominantColors(cropped)
        #     hsl_value_dominant = colorsys.rgb_to_hsv(
        #         dominantColors[2]/255, dominantColors[1]/255, dominantColors[0]/255)
        #     hsl_value_average = colorsys.rgb_to_hsv(
        #         dominantColors[2]/255, averageColors[1]/255, dominantColors[0]/255)
        #     # for para comparar resultados com os outros
        #     for index2 in List_Index_Object_IDs.keys():
        #         # Definir metricas para ser o mesmo ID

        #         for key, value in colors.ColorRangeHue.items():
        #             if value[0] <= int(hsl_value_average[0]*60) <= value[1]:
        #                 print(index, "Average", key, int(
        #                     hsl_value_average[0]*60))
        #                 break
        #         for key, value in colors.ColorRangeHue.items():
        #             if value[0] <= int(hsl_value_dominant[0]*60) <= value[1]:
        #                 print(index, "Dominant", key, int(
        #                     hsl_value_dominant[0]*60))
        #                 break
        ############################################

        # MATCH_THRESHOLD = 0.8
        # if len(List_Index_Object_IDs) > 0:
        #     for Index in List_Index_Object_IDs:
        #         (ymin, xmin, ymax, xmax) = detections['detection_boxes'][Index]
        #         (x, y) = get_center_box(ymin, xmin, ymax, xmax, width, height)

        #         if CenterPosition.get(Index) is not None:
        #             if len(CenterPosition.get(Index)) == NUM_POINTS_TRACKING:
        #                 CenterPosition.get(Index).pop(0)
        #                 CenterPosition.get(Index).append((x, y))
        #             else:
        #                 CenterPosition.get(Index).append((x, y))
        #         else:
        #             CenterPosition.__setitem__(Index, [(x, y)])

        #         # Draw all the points
        #         for index, (_x, _y) in enumerate(CenterPosition.get(Index)):
        #             diff_color = 255-index*int(255/NUM_POINTS_TRACKING)
        #             cv2.circle(image_np_with_detections,
        #                        (int(_x), int(_y)), 10, (diff_color, diff_color, 255), -1)

        #         # ####### Template Matching #######
        #         # if oldBox is not None:
        #         #     gray = cv2.cvtColor(
        #         #         image_np, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        #         #     # (left, right, top, bottom) = oldBox
        #         #     # (left, right, top, bottom) = (xmin * width, xmax * width,
        #         #     #                             ymin * height, ymax * height)
        #         #     # image_np[int(left):int(right), int(top):int(bottom)]
        #         #     # rect = image_np[int(left):int(right), int(top):int(bottom)]
        #         #     # print(image_np.shape)
        #         #     # print(rect.shape)
        #         #     rect = cv2.cvtColor(
        #         #         oldBox, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        #         #     # w = int(ymax)-int(ymin)
        #         #     # h = int(xmax) - int(xmin)
        #         #     h,w = rect.shape[::-1]
        #         #     methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
        #         #                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
        #         #     method = eval(methods[1])

        #         #     res = cv2.matchTemplate(gray, rect, method)

        #         #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #         #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #         #         top_left = min_loc
        #         #     else:
        #         #         top_left = max_loc

        #         #     bottom_right = (top_left[0] + w, top_left[1] + h)
        #         #     print(bottom_right, top_left)
        #         #     print(left, right, top, bottom)
        #         #     cv2.rectangle(image_np_with_detections,
        #         #                   top_left, bottom_right, 255, 2)
        #         # #################################
        #         # (left, right, top, bottom) = (xmin * width, xmax * width,
        #         #                               ymin * height, ymax * height)
        #         # oldBox = image_np[int(left):int(right), int(top):int(bottom)]

        #         ####### Interseta #######
        #         (DoIntersect, orientacao) = intersect.doIntersect(CenterPosition.get(Index)[len(CenterPosition.get(
        #             Index))-2], CenterPosition.get(Index)[len(CenterPosition.get(Index))-1], start_point, end_point)
        #         if DoIntersect:
        #             if orientacao == 0:
        #                 print("Collinear", Index)
        #             if orientacao == 1:
        #                 print("Esquerda", Index)
        #             if orientacao == 2:
        #                 print("Direita", Index)
        #         #########################

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            (detections['detection_classes'] + label_id_offset).astype(int),
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=MIN_SCORE_THRESH,
            # TRACKID Fazer depois
            track_ids=tracked_ids,
            agnostic_mode=False,
            # keypoints=keypoints,
            # keypoint_scores=keypoint_scores,
            # keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
        )
        ####### Frames #######
        fps = 1/(new_frame_time-prev_frame_time)
        # print(fps)
        prev_frame_time = new_frame_time
        cv2.putText(image_np_with_detections, str(int(fps)), (7, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
        ######################

        color = (0, 255, 0)
        thickness = 10
        cv2.line(image_np_with_detections, start_point,
                 end_point, color, thickness)

        cv2.imshow('object tracking',  cv2.resize(
            image_np_with_detections, (width, height)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
