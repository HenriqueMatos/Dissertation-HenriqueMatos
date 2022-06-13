import datetime
import cv2
from matplotlib.pyplot import flag
import numpy as np
import imutils

import TensorFlowObjectTracking.Class_ID_Association
import centroidtracker

MIN_SCORE_THRESH = 0.5


def load_yolo():
    # net = cv2.dnn.readNet("yolo/yolov3-tiny.weights", "yolo/yolov3-tiny.cfg")
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    classes = []
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    output_layers = [
        layer_name for layer_name in net.getUnconnectedOutLayersNames()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    print(classes)
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    # print(outputs)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            # print(conf)
            if conf > MIN_SCORE_THRESH:
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


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    # print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            # print(i, confs[i])
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)


# def hungarianAlgorith():


if __name__ == '__main__':
    ct = centroidtracker.CentroidTracker()
    id_tracker = Class_ID_Association.ID_Tracker()
    model, classes, colors, output_layers = load_yolo()
    # cap = cv2.VideoCapture('/dev/video2')
    # cap = cv2.VideoCapture("./output.mp4")
    cap = cv2.VideoCapture("./ch01_08000000058000601.mp4")
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    while True:
        _, frame = cap.read()
        # frame = imutils.resize(frame, width=800)
        # frame = frame.resize((640,360))
        total_frames = total_frames + 1
        height, width, channels = frame.shape
        # print(height, width)
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)

        # DRAW LABELS
        indexes = cv2.dnn.NMSBoxes(boxes, confs, MIN_SCORE_THRESH, 0.4)
        # boxes_final = [boxes[i] for i in indexes]
        # class_ids_final = [class_ids[i] for i in indexes]
        boxes_final = []
        class_ids_final = []
        for i in indexes:
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
        tracked_ids = id_tracker.updateData(boxes2)
        print("AQUI", len(centroid_boxes))
        # print(len(value.keys()), len(tracked_ids))
        Exist_indexs = []
        for centroid in centroid_boxes:
            for key, item in value.items():
                if np.array_equal(item, centroid):
                    Exist_indexs.append(key)
        # print(len(boxes2), len(Exist_indexs), len(tracked_ids))
        # print("AQUI", Exist_indexs, tracked_ids)
        print("AQUI2", len(boxes2), len(value))
        for index, (tracked_id, box, class_id, centroid) in enumerate(zip(tracked_ids, boxes_final, class_ids_final, centroid_boxes)):
            label = ""
            flag = False
            # print(i, confs[i])
            x, y, w, h = box
            # if (x, y, x+w, y+h) in list(value.values()):
            # id = -1
            if class_id == 0:
                # cv2.putText(frame, str(tracked_id), (centroid[0] + 10, centroid[1] + 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(
                    frame, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)

                for key, item in value.items():
                    if np.array_equal(item, centroid):
                        # label = str(key)+" "
                        id = key
                        cv2.putText(frame, str(id), (centroid[0] - 10, centroid[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        flag = True
                        break
                # if flag == False:
                #     print("Deumerda",centroid)
                #     sys.exit(0)
                # label = str(list(value.keys())
                #             [list(value.values()).index(centroid)])+" "

            label = label + str(classes[class_id])
            color = colors[tracked_id]
            print("AQUI", (x, y), (x+w, y+h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(frame, label, (x, y - 5),
            #             cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        for OldBox in id_tracker.oldBoxDetection:
            cx, cy, w, h, vx, vy, vw, vh = OldBox.kalmanfilter.predict()
            xmin = int(abs(cx-w/2))
            xmax = int(abs(cx+w/2))
            ymin = int(abs(cy-h/2))
            ymax = int(abs(cy+h/2))
            color = colors[OldBox.id]
            print((xmin, ymin), (xmax, ymax))
            # if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
            #     print("erro")
            # else:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

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
