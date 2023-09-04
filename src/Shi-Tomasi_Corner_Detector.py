import cv2
import numpy as np

class ShiTomasi:
    def __init__(self, img):
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)

    def findCorner(self):
        # (input, # of corners, threshold, minimum distance between corners)
        corners = np.float32(cv2.goodFeaturesToTrack(self.gray,25,0.01,10))
        corners = np.int0(corners)

        for i in corners:
            x,y = i.ravel()
            cv2.circle(self.img,(x,y),3,(0,0,255),-1)
            print("x",x)
            print("y",y)

        # print("x_corner",corners[:,0,0])
        # print("y_corner",corners[:,0,1])

        return self.img

img = cv2.imread('./images/oxford.jpg')

ShiTomasi = ShiTomasi(img)
img = ShiTomasi.findCorner()

# cv2.imshow('dst',img)
# cv2.waitKey()
# cv2.destroyAllWindows()