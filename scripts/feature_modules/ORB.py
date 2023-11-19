import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import DataFrameUtils, PlotUtils


class ORB:
    def __init__(self):
        self.orb = cv2.ORB_create()

    def run_module(self, input: dict) -> dict:
        img0_keypoints_df, img0_descriptors = self.run(input["img0"])
        img1_keypoints_df, img1_descriptors = self.run(input["img1"])
        return {
            "img0_keypoints": img0_keypoints_df["keypoints"],
            "img1_keypoints": img1_keypoints_df["keypoints"],
            "img0_descriptors": img0_descriptors,
            "img1_descriptors": img1_descriptors,
        }

    def run(
        self, img: np.ndarray, image_output: bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray] or Tuple[pd.DataFrame, np.ndarray]:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)

        keypoints_df = DataFrameUtils.make_data_frame_from_keypoints(keypoints)

        if image_output is True:
            img_result = cv2.drawKeypoints(
                img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            return img_result, keypoints_df, descriptors
        else:
            return keypoints_df, descriptors


if __name__ == "__main__":
    img = cv2.imread("./images/oxford.jpg", cv2.IMREAD_COLOR)

    orb = ORB()
    img_result, keypoints_df, descriptors = orb.run(img, image_output=True)

    PlotUtils.show_image(type(orb).__name__, img_result)
