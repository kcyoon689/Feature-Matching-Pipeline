import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8
from utils import DataFrameUtils, PlotUtils


class BRIEF:
    def __init__(self):
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    def run_module(self, input: dict) -> dict:
        # TODO: Implement this function
        return {}

    def run(self, img: np.ndarray, image_output: bool = False) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray] or Tuple[pd.DataFrame, np.ndarray]:
        keypoints1 = self.star.detect(img, mask=None)
        keypoints2, descriptors = self.brief.compute(img, keypoints1)

        keypoints_df = DataFrameUtils.make_data_frame_from_keypoints(
            keypoints2)

        if image_output is True:
            img_result = cv2.drawKeypoints(
                img, keypoints2, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            return img_result, keypoints_df, descriptors
        else:
            return keypoints_df, descriptors


if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)

    brief = BRIEF()
    img_result, keypoints_df, descriptors = brief.run(img, image_output=True)

    PlotUtils.show_image(type(brief).__name__, img_result)
