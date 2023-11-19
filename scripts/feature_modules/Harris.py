import cv2
import numpy as np
import pandas as pd
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import PlotUtils


class Harris:
    def run_module(self, input: dict) -> dict:
        # TODO: Implement this function
        return {}

    def run(
        self, img: np.ndarray, image_output: bool = False
    ) -> Tuple[np.ndarray, pd.DataFrame] or pd.DataFrame:
        img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.05 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        # 중심을 찾기
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # 멈추기 위한 기준을 정하고 모서리를 정제하자
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(
            img_gray, np.float32(centroids), (5, 5), (-1, -1), criteria
        )

        keypoints = []
        keypoints_x = []
        keypoints_y = []
        for x, y in corners:
            keypoints.append(cv2.KeyPoint(x, y, size=10))
            keypoints_x.append(int(np.round(x)))
            keypoints_y.append(int(np.round(y)))

        keypoints_df = pd.DataFrame(
            {"keypoints": keypoints, "x": keypoints_x, "y": keypoints_y}
        )

        if image_output is True:
            img_result = cv2.drawKeypoints(
                img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            return img_result, keypoints_df
        else:
            return keypoints_df


if __name__ == "__main__":
    img = cv2.imread("./images/oxford.jpg", cv2.IMREAD_COLOR)

    harris = Harris()
    img_result, keypoints_df = harris.run(img, image_output=True)

    PlotUtils.show_image(type(harris).__name__, img_result)
