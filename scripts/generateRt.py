import cv2
import sys
import os
import math
import numpy as np
import random
import imutils
import matplotlib.pyplot as plt
from utils import PlotUtils, TempUtils
# https://www.andre-gaschler.com/rotationconverter/
imgPath = "/home/yoonk/Desktop/pipline/images/E05_resize_10.png"
img = cv2.imread(imgPath, cv2.IMREAD_COLOR)

(imgH, imgW) = img.shape[:2]
centerX, centerY = imgW/2, imgH/2

# rotation
randAngle = random.uniform(-30, 30)
# randAngle = 30
theta = (randAngle/180.) * np.pi
rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])

# M = cv2.getRotationMatrix2D((centerX, centerY), randAngle, 1.0)
print(randAngle)
rotated = imutils.rotate_bound(img, randAngle)

#save image
cv2.imwrite("/home/yoonk/Desktop/pipline/rotated.png", rotated)
rotatedH, rotatedW = rotated.shape[:2]
print(rotatedH, rotatedW)

randTransW = int(random.uniform(-imgW*0.2, imgW*0.2))
randTransH = int(random.uniform(-imgH*0.2, imgH*0.2))
# randTransW = 2000
# randTransH = 1000
transMatrix = np.array([randTransW, randTransH])
print(transMatrix)

t = np.array([imgW/2, imgH/2])- rotMatrix @ np.array([imgW/2, imgH/2])
print(t)
finalTMatrix = transMatrix - t
print(finalTMatrix)

paddingW = int(abs(finalTMatrix[0]))
paddingH = int(abs(finalTMatrix[1]))

paddedImg = np.zeros((rotatedH + 2*paddingH, rotatedW + 2*paddingW, 3), dtype=np.uint8)
paddedImg[paddingH:paddingH+rotatedH, paddingW:paddingW+rotatedW] = np.copy(rotated)

cv2.imwrite("/home/yoonk/Desktop/pipline/paddedImg.png", paddedImg)

print(imgW/2, imgH/2)
print(t)

shifted = imutils.translate(paddedImg, finalTMatrix[0], finalTMatrix[1])
cv2.imwrite("/home/yoonk/Desktop/pipline/translation.png", shifted)

print("===============================================")
print("randAngle: ", randAngle)
print("rotMatrix: ", rotMatrix)
print("transMatrix: ", transMatrix)
