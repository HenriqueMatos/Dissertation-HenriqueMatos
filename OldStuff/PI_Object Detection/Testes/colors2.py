from traceback import print_tb
from cv2 import getBuildInformation
from joblib import parallel_backend
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io

# img = io.imread('./cropPessoas.jpg')
# [:, :, :-1]
img = io.imread('https://i.stack.imgur.com/DNM65.png')[:, :, :-1]


def getAverageColor(image):
    return image.mean(axis=0).mean(axis=0)


def getDominantColors(image):
    image = image.reshape(-1, 3).astype(np.int8)
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        image, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))

    # Resultado em BGR
    data = {}
    for i in range(len(freqs) - 1):
        tuple = (palette[indices[i]][0], palette[indices[i]]
                 [1], palette[indices[i]][2])
        data[i] = {
            "color": tuple,
            "freqs": round(abs(freqs[i]-freqs[i+1]), 2)
        }
    return data


def getPercentageFromColors(ColorTuple1, ColorTuple2):
    # RGB
    if len(ColorTuple1) == 3 and len(ColorTuple2) == 3:
        r, g, b = ColorTuple1
        p1 = (r / 255) * 100
        p2 = (g / 255) * 100
        p3 = (b / 255) * 100
        perc1 = round((p1 + p2 + p3) / 3, 2)

        r, g, b = ColorTuple2
        p1 = (r / 255) * 100
        p2 = (g / 255) * 100
        p3 = (b / 255) * 100
        perc2 = round((p1 + p2 + p3) / 3, 2)

        return abs(perc1-perc2)
    # YUV - only UV
    if len(ColorTuple1) == 2 and len(ColorTuple2) == 2:
        u, v = ColorTuple1
        p1 = (u / 255) * 100
        p2 = (v / 255) * 100
        perc1 = round((p1 + p2) / 2, 2)

        u, v = ColorTuple1
        p1 = (u / 255) * 100
        p2 = (v / 255) * 100
        perc2 = round((p1 + p2) / 2, 2)
        return abs(perc1-perc2)


def unique_count_app(a):
    colors, count = np.unique(
        a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    # print(np.argpartition(count, -4)[-5:])
    # print(count.argmax())
    # for item in np.argpartition(count, -4)[-5:]:
    #     print(colors[item])
    return colors[count.argmax()]


def bincount_app(a):
    a2D = a.reshape(-1, a.shape[-1]).astype(np.uint8)
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    print(np.bincount(a1D).argmax())
    print("aqui", a1D)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


pixels = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
average = pixels.mean(axis=0).mean(axis=0)
print(average)
pixels = np.array(pixels.reshape(-1, 3)).astype(np.float32)
# pixels = np.float32(img.reshape(-1, 3))

# pixels = pixels.reshape(-1, 3).astype(np.float32)
n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)

dominant = palette[np.argmax(counts)]


avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

indices = np.argsort(counts)[::-1]
freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
rows = np.int_(img.shape[0]*freqs)
# print(freqs)
# print(rows)
# print(palette)
dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)

r = 255
g = 255
b = 255

p1 = (r / 255) * 100
p2 = (g / 255) * 100
p3 = (b / 255) * 100

perc1 = round((p1 + p2 + p3) / 3, 2)

values = {}
print(freqs)
print(palette)
for i in range(len(rows) - 1):
    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    # print(palette[indices[i]])
    for index, value in enumerate(palette[indices[i]]):
        # print(index,value)
        values[index] = (value/255)*100
    perc2 = round((values[0] + values[1] + values[2]) / 3, 2)
    # print(perc2)
    # print(abs(perc1-perc2))

    print(type(palette[indices[i]]), palette[indices[i]], freqs[i])


# cv2.imshow('object tracking', pixels)
# cv2.imshow('object tracking1', avg_patch)
# cv2.imshow('object tracking2', dom_patch)

# print(getDominantColors(pixels))
print(unique_count_app(pixels))


# while True:
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
