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


MIN_SCORE_THRESH = 0.5


def main(args):
    cap = cv2.VideoCapture('/dev/video2')

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
    hub_model = hub.load(
        "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
    print('model loaded!')

    while cap.isOpened():
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = image_np.reshape(
            (1, height, width, 3)).astype(np.uint8)

        detections = hub_model(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        detections['num_detections'] = num_detections

        label_id_offset = 0
        image_np_with_detections = image_np.copy()

        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0]
            keypoint_scores = detections['detection_keypoint_scores'][0]

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            (detections['detection_classes'] + label_id_offset).astype(int),
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=50,
            min_score_thresh=MIN_SCORE_THRESH,
            # TRACKID Fazer depois
            track_ids=None,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=COCO17_HUMAN_POSE_KEYPOINTS)

        cv2.imshow('object detection',  cv2.resize(
            image_np_with_detections, (width, height)))

        List_Index_Object = []
        for index, (detection_scores, detection_classes) in enumerate(zip(detections['detection_scores'], detections['detection_classes'])):
            if detection_scores >= MIN_SCORE_THRESH and detection_classes in category_index.keys():
                List_Index_Object.append(index)

        print(List_Index_Object)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
