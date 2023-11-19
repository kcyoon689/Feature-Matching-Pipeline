import cv2
import imutils
import numpy as np
import pandas as pd
from scipy.linalg import svd, det
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from feature_modules import LoFTR
from utils import PlotUtils


class SVDTF:
    def run_module(self, input: dict) -> dict:
        R, t_zero, t_center = self.run(
            input["img0_features_matched"], input["img1_features_matched"]
        )
        return {
            "rotation_matrix": R,
            "translation_matrix_zero": t_zero,
            "translation_matrix_center": t_center,
        }

    def run(
        self, feature_df0: pd.DataFrame, feature_df1: pd.DataFrame
    ) -> Tuple[np.array, np.array, np.array]:
        # 1. Finding the centroids
        feature_df0_mean = np.array([np.mean(feature_df0[1]), np.mean(feature_df0[0])])
        feature_df1_mean = np.array([np.mean(feature_df1[1]), np.mean(feature_df1[0])])

        # print(feature_df0_mean)
        # print(feature_df1_mean)

        # 2. Finding the optimal rotation
        feature_df0_unbiased = [
            feature_df0[1] - feature_df0_mean[1],
            feature_df0[0] - feature_df0_mean[0],
        ]
        feature_df1_unbiased = [
            feature_df1[1] - feature_df1_mean[1],
            feature_df1[0] - feature_df1_mean[0],
        ]

        feature_df0_unbiased_np = np.array(feature_df0_unbiased)
        feature_df1_unbiased_np = np.array(feature_df1_unbiased)

        # H = np.matmul(feature_df0_unbiased_np, feature_df1_unbiased_np.transpose())
        H = feature_df0_unbiased_np @ feature_df1_unbiased_np.transpose()
        U, s, VT = svd(H)

        R = VT @ U.transpose()
        # print(R)

        # 2.a Special reflection case
        if det(R) < 0:
            [U, S, V] = svd(R)

            # multiply 3rd column of V by -1
            # print(V)
            V[0][1] = V[0][1] * -1
            V[1][1] = V[1][1] * -1
            # print(V)

            R = V * U.transpose()
            # print(R)

        # 3. Finding the translation t
        t_zero = feature_df1_mean.transpose() - R @ feature_df0_mean.transpose()
        t_center = feature_df1_mean.transpose() - feature_df0_mean.transpose()
        # print(t_zero)
        # print(t_center)

        return R, t_zero, t_center


if __name__ == "__main__":
    img0 = cv2.imread("./images/oxford.jpg", cv2.IMREAD_COLOR)
    img1 = cv2.imread("./images/oxford2.jpg", cv2.IMREAD_COLOR)

    resize_factor = 2
    h0, w0, c0 = img0.shape
    img0_resized = cv2.resize(
        img0,
        dsize=(int(w0 / resize_factor), int(h0 / resize_factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    resize_factor = 4
    h1, w1, c1 = img1.shape
    img1_resized = cv2.resize(
        img1,
        dsize=(int(w1 / resize_factor), int(h1 / resize_factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    loftr = LoFTR()
    (
        img0_result,
        img1_result,
        img_result,
        img0_feature_df,
        img1_feature_df,
        img0_feature_matched_df,
        img1_feature_matched_df,
    ) = loftr.run(img0_resized, img1_resized, image_output=True)

    svdtf = SVDTF()
    R, t_zero, t_center = svdtf.run(img0_feature_matched_df, img1_feature_matched_df)
    print(R)
    print(t_zero)
    print(t_center)

    rotation_angle = np.arctan2(R[1][0], R[0][0])

    img0_result_rotated = imutils.rotate_bound(img0_result, rotation_angle)
    h0, w0, c0 = img0_result_rotated.shape
    print(h0, w0, c0)

    h1, w1, c1 = img1_result.shape
    print(h1, w1, c1)

    # TODO
    t_u = int(np.round(t_center[1]))  # 538
    t_v = int(np.round(t_center[0]))  # -34
    print(t_u, t_v)

    # t_u, t_v: positive
    # img0_svd_result = np.zeros(
    #     (max(t_v+h0, h1), max(t_u+w0, w1), 3), dtype=np.uint8)
    # img0_svd_result[t_v:t_v+h0, t_u:t_u+w0] = np.copy(img0_result_rotated)

    # img1_svd_result = np.zeros(
    #     (max(t_v+h0, h1), max(t_u+w0, w1), 3), dtype=np.uint8)
    # img1_svd_result[0:h1, 0:w1] = np.copy(img1_result)

    # t_u: positive, t_v: negative
    img0_svd_result = np.zeros(
        (max(h0, -t_v + h1), max(t_u + w0, w1), 3), dtype=np.uint8
    )
    img0_svd_result[0:h0, t_u : t_u + w0] = np.copy(img0_result_rotated)

    img1_svd_result = np.zeros(
        (max(h0, -t_v + h1), max(t_u + w0, w1), 3), dtype=np.uint8
    )
    img1_svd_result[-t_v : -t_v + h1, 0:w1] = np.copy(img1_result)

    img_svd_result = cv2.addWeighted(img0_svd_result, 0.5, img1_svd_result, 0.5, 0)

    PlotUtils.show_image(type(svd).__name__, img_svd_result)
