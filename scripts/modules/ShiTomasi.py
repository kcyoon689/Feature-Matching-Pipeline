import cv2
import numpy as np
import pandas as pd

class ShiTomasi:
    def run(self, img, image_output=False):
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        # (input, # of corners, threshold, minimum distance between corners)
        corners = np.float32(cv2.goodFeaturesToTrack(self.gray,25,0.01,10))
        corners = np.int0(corners)

        x_corner = []
        y_corner = []

        for i in corners:
            x,y = i.ravel()
            x_corner.append(x)
            y_corner.append(y)
            cv2.circle(self.img,(x,y),3,(0,0,255),-1)
            # print("x",x)
            # print("y",y)
        corners_pd = pd.DataFrame({'x':x_corner,'y':y_corner})
        # print("x_corner",corners[:,0,0])
        # print("y_corner",corners[:,0,1])

        if image_output is True:
            return self.img, corners_pd
        else:
            return corners_pd
        # return 

if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg')

    ShiTomasi = ShiTomasi()
    corners_pd = ShiTomasi.run(img)

    print(corners_pd)
    # cv2.imshow('dst',img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
