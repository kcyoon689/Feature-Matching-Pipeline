import cv2
import numpy as np
import pandas as pd
from .SIFT import SIFT
from .ORB import ORB

class FLANNMatcher:
    def run(self, query_img, train_img, query_keypoints, train_keypoints, query_descriptor, train_descriptor, image_output=False):
        # 인덱스 파라미터 설정 ---①
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
        # 검색 파라미터 설정 ---②
        search_params=dict(checks=32)
        # Flann 매처 생성 ---③
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # 매칭 계산 ---④
        matches = matcher.match(query_descriptor, train_descriptor)

        matched_keypoints = []
        for i in matches:
            matched_keypoints.append(i.trainIdx)

        matched_keypoints_df = pd.DataFrame(matched_keypoints)

        result_img = cv2.drawMatches(query_img, query_keypoints, train_img, train_keypoints, matches, None, \
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        if image_output is True:
            return result_img, matched_keypoints_df
        else:
            return matched_keypoints_df

if __name__ == "__main__":

    query_img = cv2.imread('./images/oxford.jpg')
    train_img = cv2.imread('./images/oxford2.jpg')
    gray1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = ORB()
    query_output = orb.run(query_img)
    train_output = orb.run(train_img)
    query_keypoints, query_descriptor = orb.run(query_img)
    train_keypoints, train_descriptor = orb.run(train_img)

    FLANNMatcher = FLANNMatcher()
    result = FLANNMatcher.run(query_img, train_img, query_keypoints, train_keypoints, query_descriptor, train_descriptor)
    # 매칭 그리기

    # 결과 출력
    window_name = "BF with SIFT"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2000, 1500)
    cv2.imshow(window_name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()