import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8
from feature_modules import ORB
from utils import PlotUtils


class BFMatcher:
    def __init__(self):
        self.bf = cv2.BFMatcher()
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def run_module(self, input: dict) -> dict:
        # TODO: Implement this function
        return {}

    def run(self, img0, img1: np.ndarray, img0_keypoints, img0_descriptor: np.ndarray, img1_keypoints,
            img1_descriptor: np.ndarray, image_output=False) -> Tuple[np.ndarray, pd.DataFrame] or pd.DataFrame:
        # # 디스크립터들 매칭시키기
        img0_descriptor = np.array(img0_descriptor, dtype=np.uint8)
        img1_descriptor = np.array(img1_descriptor, dtype=np.uint8)
        matches = self.bf.knnMatch(img0_descriptor, img1_descriptor, k=2)

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

        if image_output is True:
            # # flags=2는 일치되는 특성 포인트만 화면에 표시!
            img_result = cv2.drawMatchesKnn(
                img0, img0_keypoints, img1, img1_keypoints, good, None, flags=2)

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

    bfMatcher = BFMatcher()
    img_result, matched_keypoints_df = bfMatcher.run(
        img0_result, img1_result, img0_keypoints_df['keypoints'], img0_descriptors, img1_keypoints_df['keypoints'], img1_descriptors, image_output=True)

    PlotUtils.show_image(type(bfMatcher).__name__, img_result)
