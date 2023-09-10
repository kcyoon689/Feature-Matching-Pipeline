import cv2
import numpy as np
import pandas as pd

class Harris:
    def run(self, img, image_output=False):
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.gray = np.float32(self.gray)
        self.dst = cv2.cornerHarris(self.gray,2,3,0.04)
        self.dst = cv2.dilate(self.dst,None)
        ret, self.dst = cv2.threshold(self.dst,0.05*self.dst.max(),255,0)
        self.dst = np.uint8(self.dst)

        # 중심을 찾기
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(self.dst)

        # 멈추기 위한 기준을 정하고 모서리를 정제하자
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 ,0.001)
        corners = cv2.cornerSubPix(self.gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        # 그리자!
        res = np.hstack((centroids,corners))
        res = np.int0(res)

        # res.shape=(n,4) n은 코너의 개수
        # 각 행은 [x1,y1,x2,y2]로 구성되어 있음
        # 각 행의 0,1번째 열은 중심점 좌표
        # 각 행의 2,3번째 열은 모서리 좌표
        # x 좌표: res[:,0], res[:,2]
        # y 좌표: res[:,1], res[:,3]

        keypoints1_x = []
        keypoints1_y = []
        keypoints2_x = []
        keypoints2_y = []

        for i in res[:,0]:
            keypoints1_x.append(i)
        for i in res[:,1]:
            keypoints1_y.append(i)
        for i in res[:,2]:
            keypoints2_x.append(i)
        for i in res[:,3]:
            keypoints2_y.append(i)
        
        keypoints1_pd = pd.DataFrame({'x':keypoints1_x,'y':keypoints1_y})
        keypoints2_pd = pd.DataFrame({'x':keypoints2_x,'y':keypoints2_y})
        # 각 열만 추출
        # img[res[:,1],res[:,0]] = [0,0,255]
        self.img[res[:,3],res[:,2]] = [0,255,0]

        # return print("x",res[:,2])
        if image_output is True:
            return self.img, keypoints1_pd, keypoints2_pd
        else:
            return keypoints1_pd, keypoints2_pd
        

if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg')

    harris = Harris()
    img, keypoints1_pd, keypoints2_pd = harris.run(img)
    print(keypoints1_pd)