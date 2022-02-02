import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

cap = cv2.VideoCapture('/dev/video2')

while True:

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

    image_pil = Image.fromarray(np.uint8(image_np))
    draw = ImageDraw.Draw(image_pil)

    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=4,
              fill="red")

    (x, y) = get_center_box(box, im_width, im_height)
    diff = 10
    draw.ellipse((x-diff, y-diff, x+diff, y+diff), fill=(255, 0, 0))
    draw.point(get_center_box(box, im_width, im_height), fill='red')

    # image_np = np.float32(image_np[int(xmin):int(ymin), int(xmax):int(ymin)])
    image_np = np.float32(image_np)
    # pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        image_np, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    print(dominant)

    cv2.imshow('object tracking',  cv2.resize(
        dominant, (im_width, im_height)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# image_pil.show()
