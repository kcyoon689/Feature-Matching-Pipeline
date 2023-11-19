import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from feature_modules import ORB
from utils import PlotUtils


class BFMatcher:
    def __init__(self):
        self.bf = cv2.BFMatcher()
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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
        # # 디스크립터들 매칭시키기
        img0_descriptor = np.array(img0_descriptor, dtype=np.uint8)
        img1_descriptor = np.array(img1_descriptor, dtype=np.uint8)
        matches = self.bf.knnMatch(img0_descriptor, img1_descriptor, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)

        # ratio test 적용
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img0_feature_matched = [[], []]
        img1_feature_matched = [[], []]
        for i in good:
            img0_feature_matched[0].append(img0_keypoints[i[0].queryIdx].pt[0])
            img0_feature_matched[1].append(img0_keypoints[i[0].queryIdx].pt[1])
            img1_feature_matched[0].append(img0_keypoints[i[0].trainIdx].pt[0])
            img1_feature_matched[1].append(img0_keypoints[i[0].trainIdx].pt[1])
        img0_feature_matched_df = pd.DataFrame(img0_feature_matched)
        img1_feature_matched_df = pd.DataFrame(img1_feature_matched)

        if image_output is True:
            img_result = cv2.drawMatchesKnn(
                img0,
                img0_keypoints,
                img1,
                img1_keypoints,
                good,
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

    bfMatcher = BFMatcher()
    img_result, img0_feature_matched_df, img1_feature_matched_df = bfMatcher.run(
        img0_result,
        img1_result,
        img0_keypoints_df["keypoints"],
        img0_descriptors,
        img1_keypoints_df["keypoints"],
        img1_descriptors,
        image_output=True,
    )

    PlotUtils.show_image(type(bfMatcher).__name__, img_result)
