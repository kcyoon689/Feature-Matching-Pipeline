import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8
from utils import DataFrameUtils, PlotUtils


class FAST:
    def __init__(self):
        self.fast = cv2.FastFeatureDetector_create(
            threshold=100, nonmaxSuppression=True)

    def run(self, input: dict) -> dict:
        # TODO: Implement this function
        return {}

    def run(self, img: np.ndarray, image_output: bool = False) -> Tuple[np.ndarray, pd.DataFrame] or pd.DataFrame:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints = self.fast.detect(img_gray, mask=None)

        keypoints_df = DataFrameUtils.make_data_frame_from_keypoints(keypoints)

        if image_output is True:
            img_result = cv2.drawKeypoints(
                img, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            return img_result, keypoints_df
        else:
            return keypoints_df


if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)

    fast = FAST()
    img_result, keypoints_df = fast.run(img, image_output=True)

    PlotUtils.show_image(type(fast).__name__, img_result)
