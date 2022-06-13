import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('./imagemPessoas.jpg', 0)
img2 = img.copy()
template = cv.imread('./cropPessoas.jpg', 0)
w, h = template.shape[::-1]
# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

# for meth in methods:
img = img2.copy()
method = eval(methods[1])

cap = cv.VideoCapture('/dev/video2')

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:

    ret, frame = cap.read()
    image_np = np.array(frame)
    ######## Importante ########
    # cv.COLOR_BGR2YUV
    ######## Importante ########
    gray = cv.cvtColor(image_np, cv.COLOR_BGR2GRAY)
    input_tensor = gray.astype(np.float32)

    # print(image_np)

    # Apply template Matching
    templateIMG = template.astype(np.float32)
    # threshold = 0.8
    res = cv.matchTemplate(input_tensor, templateIMG, method)
    # print( res >= threshold)
    # break
    # loc = np.where( res >= threshold)
    # if res >= threshold:

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(image_np, top_left, bottom_right, 255, 2)

    # cv.imshow('Matching Result', res)
    cv.imshow('Detected Result', image_np)

    if cv.waitKey(10) & 0xFF == ord('q'):
        cv.destroyAllWindows()
        break
