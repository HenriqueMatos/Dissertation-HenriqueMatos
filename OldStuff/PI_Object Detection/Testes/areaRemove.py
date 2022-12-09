import numpy
import cv2 as cv2
from PIL import Image, ImageDraw

# read image as RGB and add alpha (transparency)
im = Image.open("./imagemPessoas.jpg").convert("RGBA")

# convert to numpy (for convenience)
imArray = numpy.asarray(im)

## create mask
# polygon = [(444,203),(623,243),(691,177),(581,26),(482,42)]
# maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
# ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
# mask = numpy.array(maskIm)

# # assemble new image (uint8: 0-255)
# newImArray = numpy.empty(imArray.shape,dtype='uint8')

# # colors (three first columns, RGB)
# newImArray[:,:,:3] = imArray[:,:,:3]

# # transparency (4th column)
# newImArray[:,:,3] = mask*255

# newImArray[:,:,3] = (1-mask)*255

pts = numpy.array([[12,45], [34,56], [23,76], [67,98], [93,56]])
isClosed = False
color = (34,65,200)
thickness = 5

cv2.polylines(imArray, [pts], isClosed, color, thickness)




# back to Image from numpy
newIm = Image.fromarray(imArray, "RGBA")
newIm.save("out.png")