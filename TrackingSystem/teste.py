import cv2
cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
im = cv2.imread("LEVANTAMENTO UM-Model-1.png")                    # Read image
# imS = cv2.resize(im, (119350, 91800))                # Resize image
cv2.imshow("output", im)                       # Show image
cv2.waitKey(0)   