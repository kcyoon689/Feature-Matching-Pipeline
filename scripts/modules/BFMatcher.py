import cv2
import pandas as pd
from SIFT import SIFT
from ORB import ORB

class BFMatcher:
    def __init__(self):
        self.bf = cv2.BFMatcher()
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)


    def run(self, query_img, train_img, query_keypoints, train_keypoints, query_descriptor, train_descriptor):
        # # 디스크립터들 매칭시키기
        matches = self.bf.knnMatch(query_descriptor, train_descriptor,k=2)
        # matches = self.bf.match(query_descriptor, train_descriptor)

        # ratio test 적용
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        # # 거리에 기반하여 순서 정렬하기
        # matches = sorted(matches, key = lambda x:x.distance)
        matched_keypoints = []
        for i in good:
            matched_keypoints.append(i[0].trainIdx)
        matched_keypoints_df = pd.DataFrame(matched_keypoints)

        # # flags=2는 일치되는 특성 포인트만 화면에 표시!
        # result = cv2.drawMatches(query_img,query_keypoints,train_img,train_keypoints,matches[:10],None,flags=2)
        result = cv2.drawMatchesKnn(query_img, query_keypoints, train_img, train_keypoints,good,None,flags=2)

        return result, matched_keypoints_df

if __name__ == "__main__":
    query_img = cv2.imread('/home/yoonk/pipline/images/oxford.jpg', cv2.IMREAD_COLOR)
    train_img = cv2.imread('/home/yoonk/pipline/images/oxford2.jpg', cv2.IMREAD_COLOR)

    # sift = SIFT()
    # query_img, query_keypoints, query_descriptor = sift.run(query_img, image_output=True)
    # train_img, train_keypoints, train_descriptor = sift.run(train_img, image_output=True)

    orb = ORB()
    query_img, query_keypoints, query_descriptor = orb.run(query_img, image_output=True)
    train_img, train_keypoints, train_descriptor = orb.run(train_img, image_output=True)

    bfMatcher = BFMatcher()
    result_sift, matched_keypoints_df = bfMatcher.run(query_img, train_img, query_keypoints, train_keypoints, query_descriptor, train_descriptor)

    window_name = "BF with SIFT"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2000, 1500)
    cv2.imshow(window_name, result_sift)
    cv2.waitKey(0)
    cv2.destroyAllWindows()