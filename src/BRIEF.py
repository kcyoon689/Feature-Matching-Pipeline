import cv2
import numpy as np

class BRIEF:
    def __init__(self, img):
        self.img = img
        # self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.img2 = None
        # self.gray = np.float32(self.gray)
    
    def findCorner(self):
        # STAR 탐지기 먼저 개시
        star = cv2.xfeatures2d.StarDetector_create()

        # BRIEF 추출기 개시
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # STAR로 키포인트를 검출하고 BRIEF로 디스크립터 계산
        kp1 = star.detect(self.img,None)
        kp2,des = brief.compute(self.img,kp1)

        for i in kp1:
            x,y = i.pt
            print("x",x)
            print("y",y)
        
        self.img2 = cv2.drawKeypoints(self.img,kp1,self.img2,(255,0,0))

        return self.img2

img1 = cv2.imread('./images/oxford.jpg')

brief = BRIEF(img1)
img1 = brief.findCorner()

# cv2.imshow('dst',img1)
# cv2.waitKey()
# cv2.destroyAllWindows()