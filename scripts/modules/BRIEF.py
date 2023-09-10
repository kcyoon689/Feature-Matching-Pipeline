import cv2
import numpy as np
import pandas as pd
class BRIEF:
    def run(self, img):
        self.img = img
        self.img2 = None
        # STAR 탐지기 먼저 개시
        star = cv2.xfeatures2d.StarDetector_create()

        # BRIEF 추출기 개시
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # STAR로 키포인트를 검출하고 BRIEF로 디스크립터 계산
        keypoints1 = star.detect(self.img,None)
        keypoints2,descriptor = brief.compute(self.img,keypoints1)

        keypoints1_x = []
        keypoints1_y = []

        keypoints2_x = []
        keypoints2_y = []

        for i in keypoints1:
            x,y = i.pt
            keypoints1_x.append(x)
            keypoints1_y.append(y)
            # print("x",x)
            # print("y",y)
        
        for j in keypoints2:
            x,y = j.pt
            keypoints2_x.append(x)
            keypoints2_y.append(y)

        # keypoints1_df = pd.DataFrame({'x':keypoints1_x,'y':keypoints1_y})
        keypoints_df = pd.DataFrame({'x':keypoints2_x,'y':keypoints2_y})

        self.img2 = cv2.drawKeypoints(self.img,keypoints2,self.img2,(255,0,0))
        
        return self.img2, keypoints_df

if __name__ == "__main__":
    img1 = cv2.imread('./images/oxford.jpg')

    brief = BRIEF()
    result_img, keypoints_df = brief.run(img1)
    
    cv2.imshow('dst',result_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
