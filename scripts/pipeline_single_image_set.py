import cv2

from modules.SIFT import SIFT
from modules.BFMatcher import BFMatcher

# flag from python arg
flag_sift = True
flag_bf_matcher = True

query_img = cv2.imread('/home/yoonk/pipline/images/oxford.jpg', cv2.IMREAD_COLOR)
train_img = cv2.imread('/home/yoonk/pipline/images/oxford2.jpg', cv2.IMREAD_COLOR)

# 1. Feature Extractor
# 2. Feature Descriptor
if flag_sift is True:
    sift = SIFT()
    query_keypoints, query_descriptor = sift.run(query_img)
    train_keypoints, train_descriptor = sift.run(train_img)

# if flag_orb is True:
#     orb = ORB()
#     query_keypoints, query_descriptor = orb.run(query_img)
#     train_keypoints, train_descriptor = orb.run(train_img)

# 3. Feature Matcher
if flag_bf_matcher is True:
    bf_matcher = BFMatcher()
    result_sift = bf_matcher.run(query_img, train_img, 
                                 query_keypoints, train_keypoints, 
                                 query_descriptor, train_descriptor)


# window_name = "SIFT: " + flag_sift + ", BFMatcher: " + flag_bf_matcher
window_name = "BF with SIFT"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 2000, 1500)
cv2.imshow(window_name, result_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()
