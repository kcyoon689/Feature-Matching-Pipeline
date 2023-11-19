import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from feature_modules import ORB
from utils import PlotUtils


class FLANNMatcher:
    def __init__(self):
        # 인덱스 파라미터 설정 ---①
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )
        # 검색 파라미터 설정 ---②
        search_params = dict(checks=32)
        # Flann 매처 생성 ---③
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def run_module(self, input: dict) -> dict:
        img0_feature_matched_df, img1_feature_matched_df = self.run(
            None,
            None,
            input["img0_keypoints"],
            input["img0_descriptors"],
            input["img1_keypoints"],
            input["img1_descriptors"],
        )
        return {
            "img0_features": None,
            "img1_features": None,
            "img0_features_matched": img0_feature_matched_df,
            "img1_features_matched": img1_feature_matched_df,
        }

    def run(
        self,
        img0,
        img1: np.ndarray,
        img0_keypoints,
        img0_descriptor: np.ndarray,
        img1_keypoints,
        img1_descriptor: np.ndarray,
        image_output=False,
    ) -> Tuple[np.ndarray, pd.DataFrame] or pd.DataFrame:
        # 매칭 계산 ---④
        matches = self.matcher.match(img0_descriptor, img1_descriptor)
        # matches = sorted(matches, key=lambda x: x.distance)

        img0_feature_matched = [[], []]
        img1_feature_matched = [[], []]
        for i in matches:
            img0_feature_matched[0].append(img0_keypoints[i.queryIdx].pt[0])
            img0_feature_matched[1].append(img0_keypoints[i.queryIdx].pt[1])
            img1_feature_matched[0].append(img0_keypoints[i.trainIdx].pt[0])
            img1_feature_matched[1].append(img0_keypoints[i.trainIdx].pt[1])
        img0_feature_matched_df = pd.DataFrame(img0_feature_matched)
        img1_feature_matched_df = pd.DataFrame(img1_feature_matched)

        if image_output is True:
            img_result = cv2.drawMatches(
                img0,
                img0_keypoints,
                img1,
                img1_keypoints,
                matches,
                None,
                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
            )

            return img_result, img0_feature_matched_df, img1_feature_matched_df
        else:
            return img0_feature_matched_df, img1_feature_matched_df


if __name__ == "__main__":
    # img0 = cv2.imread("./images/oxford.jpg", cv2.IMREAD_COLOR)
    # img1 = cv2.imread("./images/oxford2.jpg", cv2.IMREAD_COLOR)
    img0 = cv2.imread("./images/E05_resize_10.png", cv2.IMREAD_COLOR)
    img1 = cv2.imread("./images/E07_resize_10.png", cv2.IMREAD_COLOR)

    orb = ORB()
    img0_result, img0_keypoints_df, img0_descriptors = orb.run(img0, image_output=True)
    img1_result, img1_keypoints_df, img1_descriptors = orb.run(img1, image_output=True)

    FLANNMatcher = FLANNMatcher()
    img_result, img0_feature_matched_df, img1_feature_matched_df = FLANNMatcher.run(
        img0_result,
        img1_result,
        img0_keypoints_df["keypoints"],
        img0_descriptors,
        img1_keypoints_df["keypoints"],
        img1_descriptors,
        image_output=True,
    )

    PlotUtils.show_image(type(FLANNMatcher).__name__, img_result)
