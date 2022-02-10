import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

cap = cv2.VideoCapture('/dev/video2')


def unique_count_app(a):
    colors, count = np.unique(
        a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def bincount_app(a):
    a2D = a.reshape(-1, a.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


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


while True:

    ret, frame = cap.read()
    image_np = np.array(frame)
    # print(image_np)

    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    box = [0.04039052, 0.21960801, 0.7118546, 0.9326711]

    # box = [
    #     4.64870423e-01,
    #     1.80075169e-02,
    #     1.00000000e+00,
    #     7.47812033e-01
    # ]

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
    # image_np = np.float32(image_np)
    # pixels = np.float32(img.reshape(-1, 3))

    print(unique_count_app(image_np))

    print(bincount_app(image_np))

    # indices = np.argsort(counts)[::-1]
    # freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    # rows = np.int_(img.shape[0]*freqs)

    # dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    # for i in range(len(rows) - 1):
    #     dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    # image= cv2.imread('Waterfall.png')
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    print((left, right, top, bottom))
    print(im_width, im_height)
    cropped = image_np[int(left):int(right), int(top):int(bottom)]
    # cropped = image_np[int(left):int(right), int(top):int(bottom)]
    # cropped = image_np[30:250, 100:230]
    cv2.imshow('Cropped Image', cropped)

    # cv2.imshow('object tracking',  cv2.resize(
    #     dominant, (im_width, im_height)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# image_pil.show()
