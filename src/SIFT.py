# SIFT로 특징점 및 디스크립터 추출(desc_sift.py)

import cv2
import numpy as np

img = cv2.imread('./images/oxford.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT 추출기 생성
sift = cv2.xfeatures2d.SIFT_create()

# 키 포인트 검출과 서술자 계산
# kp = sift.detect(gray, None)
# keypoints = np.array(keypoints)
# descriptor = sift.detectAndCompute(gray, None)[1]
keypoints, descriptor = sift.detectAndCompute(gray, None)

for i in keypoints:
    x,y = i.pt
    print("x",x)
    print("y",y)

# print("descriptor",descriptor)
# # print("descriptor.shape",descriptor.shape)
# print("keypoints[0]",keypoints[0].pt)
# print("descriptor[0]",descriptor[0])
# print("descriptor[0].shape",descriptor[0].shape)

# print('keypoint:',len(keypoints), 'descriptor:', descriptor.shape)

# print("2")
# print(keypoints.shape)
# print(keypoints[0].pt)
# print(keypoints[0].pt[0])
# print(keypoints[0].pt[1])
# print(keypoints[0].size)
# print(keypoints[0].angle)
# print(keypoints[0].response)
# print(keypoints[0].octave)
# print(keypoints[0].class_id)


# # 키 포인트 그리기
# img_draw = cv2.drawKeypoints(img, keypoints, None, \
#                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# # 결과 출력
# cv2.imshow('SIFT', img_draw)
# cv2.waitKey()
# cv2.destroyAllWindows()