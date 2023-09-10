import cv2
import numpy as np
import pandas as pd
import utils
from typing import Tuple

from ORB import ORB


class FLANNMatcher:
    def __init__(self):
        # 인덱스 파라미터 설정 ---①
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        # 검색 파라미터 설정 ---②
        search_params = dict(checks=32)
        # Flann 매처 생성 ---③
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def run(self, input: dict) -> dict:
        # TODO: Implement this function
        return {}

    def run(self, img0, img1: np.ndarray, img0_keypoints, img0_descriptor: np.ndarray, img1_keypoints,
            img1_descriptor: np.ndarray, image_output=False) -> Tuple[np.ndarray, pd.DataFrame] or pd.DataFrame:
        # 매칭 계산 ---④
        matches = self.matcher.match(img0_descriptor, img1_descriptor)

        matched_keypoints = []
        for i in matches:
            matched_keypoints.append(i.trainIdx)

        matched_keypoints_df = pd.DataFrame(matched_keypoints)

        if image_output is True:
            img_result = cv2.drawMatches(img0, img0_keypoints, img1, img1_keypoints, matches, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            return img_result, matched_keypoints_df
        else:
            return matched_keypoints_df


if __name__ == "__main__":
    img0 = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./images/oxford2.jpg', cv2.IMREAD_COLOR)

    orb = ORB()
    img0_result, img0_keypoints_df, img0_descriptors = orb.run(
        img0, image_output=True)
    img1_result, img1_keypoints_df, img1_descriptors = orb.run(
        img1, image_output=True)

    flannMatcher = FLANNMatcher()
    img_result, matched_keypoints_df = flannMatcher.run(
        img0_result, img1_result, img0_keypoints_df['keypoints'], img0_descriptors, img1_keypoints_df['keypoints'], img1_descriptors, image_output=True)

    utils.show_image(type(flannMatcher).__name__, img_result)
