import cv2
import numpy as np
import colorsys

ColorRangeHue = {
    "red-orange": [11, 20],
    "orange-brown": [21, 40],
    "orange-yellow": [41, 50],
    "yellow": [51, 60],
    "yellow-green": [61, 80],
    "green": [81, 140],
    "green-cyan": [141, 169],
    "cyan": [170, 200],
    "cyan-blue": [201, 220],
    "blue": [221, 240],
    "blue-magenta": [241, 280],
    "magenta": [281, 320],
    "magenta-mink": [321, 330],
    "pink": [331, 345],
    "pink-red": [346, 355],
    "red": [0,360]
}
# O Red deveria ser "red": [10,355]
# mas como é o ultimo pode ter a gama toda

# BGR


def getDominantColors(a):
    colors, count = np.unique(
        a.reshape(-1, a.shape[-1]), axis=0, return_counts=True)
    # print(np.argpartition(count, -4)[-5:])
    # print(count.argmax())
    # for item in np.argpartition(count, -4)[-5:]:
    #     print(colors[item])
    return colors[count.argmax()]


def getAverageColor(image):
    return image.mean(axis=0).mean(axis=0)


def getDominantColors2(image):
    image2 = image.reshape(-1, 3).astype(np.float32)
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(
        image2, n_colors, None, criteria, 10, flags)
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
    # print(ColorTuple1, ColorTuple2)
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
