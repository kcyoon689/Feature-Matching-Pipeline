# SIFT로 특징점 및 디스크립터 추출(desc_sift.py)

import cv2
import numpy as np

class SIFT:
    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
    
    def findCorner(self):
        # SIFT 추출기 생성
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptor = sift.detectAndCompute(self.gray, None)

        for i in keypoints:
            x,y = i.pt
            # print("x",x)
            # print("y",y)

        # 키 포인트 그리기
        img_draw = cv2.drawKeypoints(self.img, keypoints, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return img_draw

img = cv2.imread('./images/oxford.jpg')

sift = SIFT(img)
img = sift.findCorner()

# cv2.imshow('dst',img)
# cv2.waitKey()
# cv2.destroyAllWindows()