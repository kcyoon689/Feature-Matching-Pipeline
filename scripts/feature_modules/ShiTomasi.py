import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8
from utils import PlotUtils


class ShiTomasi:
    def run_module(self, input: dict) -> dict:
        # TODO: Implement this function
        return {}

    def run(self, img: np.ndarray, image_output: bool = False) -> Tuple[np.ndarray, pd.DataFrame] or pd.DataFrame:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10)
        corners = (np.reshape(corners, (-1, 2)))

        corners_x, corners_y = np.split(corners, 2, axis=1)
        corners_x = corners_x.flatten()
        corners_y = corners_y.flatten()

        keypoints = []
        for x, y in zip(corners_x, corners_y):
            keypoints.append(cv2.KeyPoint(x, y, size=10))

        corners_x = np.int0(np.round(corners_x))
        corners_y = np.int0(np.round(corners_y))

        keypoints_df = pd.DataFrame({
            'keypoints': keypoints,
            'x': corners_x,
            'y': corners_y,
        })

        if image_output is True:
            img_result = cv2.drawKeypoints(
                img, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            return img_result, keypoints_df
        else:
            return keypoints_df


if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)

    shi_tomasi = ShiTomasi()
    img_result, keypoints_df = shi_tomasi.run(img, image_output=True)

    PlotUtils.show_image(type(shi_tomasi).__name__, img_result)
