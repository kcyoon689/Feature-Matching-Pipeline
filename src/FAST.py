import cv2
import numpy as np

class FAST:
    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.img2,self.img3 = None,None
    
    def findCorner(self):
        # 디폴트 값으로 FAST 객체를 시작한다
        # ()안에 임계값 넣기
        fast = cv2.FastFeatureDetector_create(100)
        
        # 키포인트를 찾고 그린다
        kp = fast.detect(self.gray,None)

        for i in kp:
            x,y = i.pt
            # print("x",x)
            # print("y",y)

        img2 = cv2.drawKeypoints(self.img,kp,self.img2,(255,0,0))

        return print("x",kp[0].pt[0])

img = cv2.imread('./images/oxford.jpg')

fast = FAST(img)
img = fast.findCorner()

# cv2.imshow('dst',img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# print(kp[0].pt)
# print("x",kp[0].pt[0])
# print("y",kp[0].pt[1])
# print(kp[1].pt)
# print("x",kp[1].pt[0])
# print("y",kp[1].pt[1])

# # NMS 사용 X
# fast.setNonmaxSuppression(0)
# kp = fast.detect(img,None)
# img3 = cv2.drawKeypoints(img,kp,img3,(255,0,0))
# cv2.imshow('FAST2',img3)

# cv2.waitKey(0)
# cv2.destroyAllWindows()