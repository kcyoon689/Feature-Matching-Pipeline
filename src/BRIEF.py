import cv2
import numpy as np

img = cv2.imread('./images/oxford.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = None

# STAR 탐지기 먼저 개시
star = cv2.xfeatures2d.StarDetector_create()

# BRIEF 추출기 개시
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# STAR로 키포인트를 검출하고 BRIEF로 디스크립터 계산
kp1 = star.detect(img,None)
kp2,des = brief.compute(img,kp1)

img2 = cv2.drawKeypoints(img,kp1,img2,(255,0,0))

cv2.imshow('Result',img2)
cv2.waitKey()
cv2.destroyAllWindows()