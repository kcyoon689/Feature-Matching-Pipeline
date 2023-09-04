import numpy as np
import cv2

img = cv2.imread('./images/oxford.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img2 = None

orb = cv2.ORB_create()

# ORB로 키포인트 / 디스크립터 찾기
kp,des = orb.detectAndCompute(img,None)

# 키포인트들의 위치만 나타낸다. 크기 /방향 x
img2 = cv2.drawKeypoints(img,kp,img2,(0,255,0),flags=0)

cv2.imshow('Res',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()