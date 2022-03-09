import datetime
import json
import cv2
from matplotlib.pyplot import flag
import numpy as np
import imutils
from simplejson import OrderedDict

# import NewClass_ID_Association
import centroidtracker
import Data_Config_Count


def load_yolo(path_model_weights, path_model_cfg, path_yolo_coco_names, wanted_classes):
    # net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
    net = cv2.dnn.readNet(path_model_weights, path_model_cfg)
    classes = []
    with open(path_yolo_coco_names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    ID_wanted_classes = []
    for className in wanted_classes:
        if className in classes:
            ID_wanted_classes.append(classes.index(className))

    output_layers = [
        layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    print(classes)
    return net, classes, colors, output_layers, ID_wanted_classes


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    # print(outputs)
    return blob, outputs


def get_box_dimensions(outputs, height, width, threshold):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            # print(conf)
            if conf > threshold:
                # print(conf)
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


if __name__ == '__main__':
    with open('config/config.json', 'r') as f:
        data = json.load(f)

    ConfigDataUpdater = Data_Config_Count.Data_Config_Count()
    ConfigDataUpdater.register(data)
    # print(ConfigDataUpdater.line_intesection_zone)

    ct = centroidtracker.CentroidTracker()
    # id_tracker = NewClass_ID_Association.ID_Tracker()
    model, classes, colors, output_layers, ID_wanted_classes = load_yolo(
        ConfigDataUpdater.path_model_weights, ConfigDataUpdater.path_model_cfg, ConfigDataUpdater.path_yolo_coco_names, ConfigDataUpdater.object_data_tracking)
    # cap = cv2.VideoCapture('/dev/video2')
    # cap = cv2.VideoCapture("./output.mp4")
    cap = cv2.VideoCapture("./ch01_08000000058000601.mp4")
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    while True:
        _, frame = cap.read()
        ####### Polygon Remove #######
        for arrayPoints in ConfigDataUpdater.remove_area:
            mask = np.zeros(frame.shape, dtype=np.uint8)
            contours = np.array(arrayPoints)
            cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
            # apply the mask
            frame = cv2.bitwise_or(frame, mask)
        ##########################

        # frame = imutils.resize(frame, width=800)
        # frame = frame.resize((640,360))
        total_frames = total_frames + 1
        height, width, channels = frame.shape
        # print(height, width)
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(
            outputs, height, width, ConfigDataUpdater.threshold)

        # DRAW LABELS
        indexes = cv2.dnn.NMSBoxes(
            boxes, confs, ConfigDataUpdater.threshold, 0.4)

        boxes_final = []
        class_ids_final = []
        for i in indexes:
            # if class_ids[i] in ID_wanted_classes:
            if class_ids[i] == 0:
                boxes_final.append(boxes[i])
                class_ids_final.append(class_ids[i])

        boxes2 = []
        centroid_boxes = []
        for item in boxes_final:
            x, y, w, h = item
            boxes2.append((x, y, x+w, y+h))
            cX = int((x + x+w) / 2.0)
            cY = int((y + y+h) / 2.0)
            centroid_boxes.append(np.asarray((cX, cY)))

        value = ct.update(boxes2)

        IDs_list = list(value.keys())
        Centroids_list = []
        for item in list(value.values()):
            Centroids_list.append(tuple(item))

        UpdateValuesCentroids = OrderedDict()
        for id in IDs_list:
            UpdateValuesCentroids[id] = ct.objectsCentroids[id]

        # Draw Zones
        for item in ConfigDataUpdater.zone:
            cv2.polylines(frame, [np.array(item["points"])],
                          True, (255, 0, 0), 2)

        # Draw Line_intersection_zone
        for item in ConfigDataUpdater.line_intersection_zone:
            cv2.line(frame, item["start_point"], item["end_point"], (0, 255, 0), 2)

        ConfigDataUpdater.updateData(UpdateValuesCentroids)

        for index, (box, class_id, centroid) in enumerate(zip(boxes_final, class_ids_final, centroid_boxes)):
            x, y, w, h = box

            id = -1
            if Centroids_list.__contains__(tuple(centroid)):
                id = IDs_list[Centroids_list.index(tuple(centroid))]

                for item in ct.objectsCentroids[id]:
                    cv2.circle(
                        frame, (item[0], item[1]), 3, (0, 255, 0), -1)

            cv2.putText(frame, str(id), (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            label = str(classes[class_id])
            color = colors[id]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)
        cv2.putText(frame, fps_text, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("Image", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
