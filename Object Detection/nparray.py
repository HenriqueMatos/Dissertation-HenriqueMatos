import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

cap = cv2.VideoCapture('/dev/video2')
ret, frame = cap.read()
image_np = np.array(frame)
# print(image_np)


im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


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


box = [
    4.64870423e-01,
    1.80075169e-02,
    1.00000000e+00,
    7.47812033e-01
]

(ymin, xmin, ymax, xmax) = box

(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                              ymin * im_height, ymax * im_height)

image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
draw = ImageDraw.Draw(image_pil)

draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
           (left, top)],
          width=4,
          fill="red")

draw.point(get_center_box(box, im_width, im_height), fill='red')

image_pil.show()
