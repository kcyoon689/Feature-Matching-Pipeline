import cv2
import numpy as np

class BRIEF:
    def run(self, img):
        self.img = img
        self.img2 = None
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

if __name__ == "__main__":
    img1 = cv2.imread('./images/oxford.jpg')

    brief = BRIEF(img1)
    img1 = brief.findCorner()

    cv2.imshow('dst',img1)
    cv2.waitKey()
    cv2.destroyAllWindows()
