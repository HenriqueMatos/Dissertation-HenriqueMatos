from black import main
import cv2
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

MIN_SCORE_THRESH = 0.4


def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return(_x, _y)


def get_center_box(box, im_width, im_height):
    (ymin, xmin, ymax, xmax) = box
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    return centroid(((left, top), (right, bottom), (left, bottom), (right, top)))


def main():
    cap = cv2.VideoCapture('/dev/video2')
    f = open("teste2_aux.txt", "w")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    PATH_TO_LABELS = './models/research/object_detection/data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(
        PATH_TO_LABELS, use_display_name=True)
    print(str(category_index))
    # new_category_index = {
    #     1: category_index[1]
    # }

    print('loading model...')
    # https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1
    # https://tfhub.dev/tensorflow/retinanet/resnet101_v1_fpn_640x640/1
    # https://tfhub.dev/tensorflow/ssd_mobilenet_v1/fpn_640x640/1
    # https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1
    print(registry.resolver(
        "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"))
    hub_model = hub.load(
        "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    # hub_model = hub.load(
    #     "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1")

    print('model loaded!')
    new_frame_time = 0
    prev_frame_time = 0
    CenterPosition = {}
    flag = True
    while cap.isOpened():
        ret, frame = cap.read()
        # Capture F
        new_frame_time = time.time()

        image_np = np.array(frame)

        input_tensor = image_np.reshape(
            (1, height, width, 3)).astype(np.uint8)

        detections = hub_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        IndexToRemove = []
        for index, (detection_scores, detection_classes) in enumerate(zip(detections['detection_scores'], detections['detection_classes'])):
            if not (detection_scores >= MIN_SCORE_THRESH and detection_classes == 1):
                IndexToRemove.append(index)
        # print(IndexToRemove)
        for key, value in detections.items():
            if type(value) == np.ndarray:
                detections[key] = np.delete(detections[key], IndexToRemove, 0)

        detections['num_detections'] = num_detections - len(IndexToRemove)

        # for key, value in detections.items():
        #     print(type(value))
        #     detections[key]=np.delete(detections[key],0,0)

        # if flag == True:
        #     f.write(str(detections))
        #     # flag = False
        # break

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        # keypoints, keypoint_scores = None, None
        # if 'detection_keypoints' in detections:
        #     keypoints = detections['detection_keypoints'][0]
        #     keypoint_scores = detections['detection_keypoint_scores'][0]

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            (detections['detection_classes'] + label_id_offset).astype(int),
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=MIN_SCORE_THRESH,
            # TRACKID Fazer depois
            track_ids=None,
            agnostic_mode=False,
            # keypoints=keypoints,
            # keypoint_scores=keypoint_scores,
            # keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS
        )

        start_point = (200, 10)
        end_point = (200, height-10)

        List_Index_Object_IDs = []
        for index, (detection_scores, detection_classes) in enumerate(zip(detections['detection_scores'], detections['detection_classes'])):
            # if detection_scores >= MIN_SCORE_THRESH and detection_classes == 1: # Person ID
            if detection_scores >= MIN_SCORE_THRESH and detection_classes in category_index.keys():
                List_Index_Object_IDs.append(index)

        NUM_POINTS = 20
        if len(List_Index_Object_IDs) > 0:
            for Index in List_Index_Object_IDs:
                box = detections['detection_boxes'][Index]
                (x, y) = get_center_box(box, width, height)

                if CenterPosition.get(Index) is not None:
                    if len(CenterPosition.get(Index)) == NUM_POINTS:
                        CenterPosition.get(Index).pop(0)
                        CenterPosition.get(Index).append((x, y))
                    else:
                        CenterPosition.get(Index).append((x, y))
                else:
                    CenterPosition.__setitem__(Index, [(x, y)])

                # Draw all the points
                for index, (_x, _y) in enumerate(CenterPosition.get(Index)):

                    diff_color = 255-index*int(255/NUM_POINTS)
                    cv2.circle(image_np_with_detections,
                               (int(_x), int(_y)), 10, (diff_color, diff_color, 255), -1)
                 # Se interseta com os dados
                if intersect.doIntersect(CenterPosition.get(Index)[len(CenterPosition.get(Index))-2], CenterPosition.get(Index)[len(CenterPosition.get(Index))-1], start_point, end_point):
                    print("Yes")
                else:
                    print("No")

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(image_np_with_detections, str(int(fps)), (7, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

        # Green color in BGR
        color = (0, 255, 0)

        # Line thickness of 9 px
        thickness = 10

        cv2.line(image_np_with_detections, start_point,
                 end_point, color, thickness)

        cv2.imshow('object tracking',  cv2.resize(
            image_np_with_detections, (width, height)))
        # print(List_Index_Object_IDs)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
