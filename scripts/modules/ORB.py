import numpy as np
import cv2

class ORB:
    def findCorner(self, img):
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.img2 = None
        # ORB 추출기 생성
        orb = cv2.ORB_create()

        # 키 포인트 검출과 서술자 계산
        kp, des = orb.detectAndCompute(self.gray,None)

        for i in kp:
            x,y = i.pt
            # print("x",x)
            # print("y",y)

        # 키 포인트 그리기
        img2 = cv2.drawKeypoints(self.img,kp,self.img2,(0,255,0),flags=0)

        return img2

if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg')

    orb = ORB(img)
    img2 = orb.findCorner()

    cv2.imshow('Res',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
