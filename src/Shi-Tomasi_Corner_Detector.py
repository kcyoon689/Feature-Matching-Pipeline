import cv2
import numpy as np

img = cv2.imread('./images/oxford.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# (input, # of corners, threshold, minimum distance between corners)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,(0,0,255),-1)

print("x",x)
print("x_corner",corners[:,0,0])

print("y",y)
print("y_corner",corners[:,0,1])

# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xFF == 27:
#     cv2.destroyAllWindows()