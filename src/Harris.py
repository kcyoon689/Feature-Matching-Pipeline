import cv2
import numpy as np

img = cv2.imread('./images/oxford.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Harris Corner 찾기
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.05*dst.max(),255,0)
dst = np.uint8(dst)

# 중심을 찾기
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# 멈추기 위한 기준을 정하고 모서리를 정제하자
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100 ,0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# for i in corners:
#     x,y = i.ravel()
#     print("x",x)
#     # print("y",y)

# 그리자!
res = np.hstack((centroids,corners))
res = np.int0(res)

# res.shape=(n,4) n은 코너의 개수
# x 좌표: res[:,0], res[:,2]
# y 좌표: res[:,1], res[:,3]

# 각 열만 추출
# img[res[:,1],res[:,0]] = [0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

print("x",res[:,2])
# cv2.imshow('dst',img)

# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()