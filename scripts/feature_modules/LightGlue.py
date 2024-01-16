import cv2
import numpy as np
import kornia as K
import kornia.feature as KF
import pandas as pd
import torch
import argparse
from kornia_moons.feature import draw_LAF_matches
from typing import Tuple

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import ComputerVisionUtils, DataFrameUtils, PlotUtils
from lib.LightGlue import lightglue


class LightGlue:
    def __init__(self, method: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = (
            lightglue.SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        )  # load the extractor
        self.matcher = lightglue.LightGlue(features=method).eval().to(self.device)
        print("LightGlue module is created successfully")

    def run_module(self, input: dict) -> dict:
        (
            img0_feature_df,
            img1_feature_df,
            img0_feature_matched_df,
            img1_feature_matched_df,
        ) = self.run(input["img0"], input["img1"])
        return {
            "img0_features": img0_feature_df,
            "img1_features": img1_feature_df,
            "img0_features_matched": img0_feature_matched_df,
            "img1_features_matched": img1_feature_matched_df,
        }

    def run(
        self, img0: np.ndarray, img1: np.ndarray, image_output=False
    ) -> (
        Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame,
        ]
        or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ):
        img0_tensor = ComputerVisionUtils.convert_cv_image_to_torch_image(img0)
        img1_tensor = ComputerVisionUtils.convert_cv_image_to_torch_image(img1)

        feats0 = self.extractor.extract(img0_tensor.to(self.device))
        feats1 = self.extractor.extract(img1_tensor.to(self.device))
        input_dict = {
            "image0": feats0,
            "image1": feats1,
        }
        matcher_result = self.matcher(input_dict)
        feats0, feats1, matcher_result = [
            lightglue.utils.rbd(x) for x in [feats0, feats1, matcher_result]
        ]  # remove batch dimension

        kpts0, kpts1, matches = (
            feats0["keypoints"],
            feats1["keypoints"],
            matcher_result["matches"],
        )
        mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
        mkpts1 = kpts1[matches[..., 1]].cpu().numpy()

        Fm, inliers = cv2.findFundamentalMat(
            mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
        )
        inliers = inliers > 0

        draw_LAF_matches(
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts0).view(1, -1, 2),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts0.shape[0]).view(1, -1, 1),
            ),
            KF.laf_from_center_scale_ori(
                torch.from_numpy(mkpts1).view(1, -1, 2),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                torch.ones(mkpts1.shape[0]).view(1, -1, 1),
            ),
            torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(img0_tensor),
            K.tensor_to_image(img1_tensor),
            inliers,
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                #    'tentative_color': (1.0, 0.5, 1),
                "tentative_color": None,
                "feature_color": (0.2, 0.5, 1),
                "vertical": True,
            },
        )

        mkpts0_df = pd.DataFrame(mkpts0)
        mkpts1_df = pd.DataFrame(mkpts1)

        filter0 = mkpts0 * inliers
        filter1 = mkpts1 * inliers

        filter0_df = pd.DataFrame(filter0)
        filter1_df = pd.DataFrame(filter1)

        img0_feature_df = DataFrameUtils.drop_zero(df=mkpts0_df)
        img1_feature_df = DataFrameUtils.drop_zero(df=mkpts1_df)
        img0_feature_matched_df = DataFrameUtils.drop_zero(df=filter0_df)
        img1_feature_matched_df = DataFrameUtils.drop_zero(df=filter1_df)

        img0_feature_matched_df = img0_feature_matched_df.reset_index(drop=True)
        img1_feature_matched_df = img1_feature_matched_df.reset_index(drop=True)

        if image_output is True:
            (
                img0_result,
                img1_result,
                img_result,
            ) = PlotUtils.plot_2d_image_with_features(
                img0=img0,
                img1=img1,
                img0_feature_df=img0_feature_df,
                img1_feature_df=img1_feature_df,
                img0_feature_matched_df=img0_feature_matched_df,
                img1_feature_matched_df=img1_feature_matched_df,
            )

            return (
                img0_result,
                img1_result,
                img_result,
                img0_feature_df,
                img1_feature_df,
                img0_feature_matched_df,
                img1_feature_matched_df,
            )
        else:
            return (
                img0_feature_df,
                img1_feature_df,
                img0_feature_matched_df,
                img1_feature_matched_df,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGlue Method")
    parser.add_argument(
        "--img0",
        type=str,
        default="images/E05_resize_10.png",
        help="path to image 0",
    )
    parser.add_argument(
        "--img1",
        type=str,
        default="images/E07_resize_10.png",
        help="path to image 1",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="superpoint",
        help="method to use",
    )

    args = parser.parse_args()

    img0 = cv2.imread(args.img0, cv2.IMREAD_COLOR)
    img1 = cv2.imread(args.img1, cv2.IMREAD_COLOR)

    resize_factor = 10
    h, w, c = img0.shape
    img0_resized = cv2.resize(
        img0,
        dsize=(int(w / resize_factor), int(h / resize_factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    resize_factor = 10
    h, w, c = img1.shape
    img1_resized = cv2.resize(
        img1,
        dsize=(int(w / resize_factor), int(h / resize_factor)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    lightGlue = LightGlue(args.method)
    (
        img0_result,
        img1_result,
        img_result,
        img0_feature_df,
        img1_feature_df,
        img0_feature_matched_df,
        img1_feature_matched_df,
    ) = lightGlue.run(img0_resized, img1_resized, image_output=True)

    PlotUtils.show_image(type(lightGlue).__name__, img_result)
