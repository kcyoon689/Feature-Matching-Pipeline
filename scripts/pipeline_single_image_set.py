import cv2

import sys
import os
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from feature_modules import BRIEF, ORB, SIFT
from feature_modules import BFMatcher, FLANNMatcher
from feature_modules import LoFTR, SVDTF
from utils import PlotUtils, TempUtils


def main(config: dict):
    # Load modules
    modules = []
    names = []
    if config["BRIEF"] is True:
        modules.append(BRIEF())
        names.append("BRIEF")

    if config["SIFT"] is True:
        modules.append(SIFT())
        names.append("SIFT")

    if config["ORB"] is True:
        modules.append(ORB())
        names.append("ORB")

    if config["LoFTR"] is True:
        modules.append(LoFTR())
        names.append("LoFTR")

    if config["BFMatcher"] is True:
        modules.append(BFMatcher())
        names.append("BFMatcher")

    if config["FLANNMatcher"] is True:
        modules.append(FLANNMatcher())
        names.append("FLANNMatcher")

    if config["SVDTF"] is True:
        modules.append(SVDTF())

    # Image preprocessing
    # img0 = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)
    # img1 = cv2.imread('./images/oxford2.jpg', cv2.IMREAD_COLOR)
    img0 = cv2.imread("./images/E05_resize_10.png", cv2.IMREAD_COLOR)
    img1 = cv2.imread("./images/E07_resize_10.png", cv2.IMREAD_COLOR)

    resize_factor0 = 10
    padding_pixel0 = int(900 / resize_factor0)
    h0, w0, c0 = img0.shape
    img0_resized = cv2.resize(
        img0,
        dsize=(int(w0 / resize_factor0), int(h0 / resize_factor0)),
        interpolation=cv2.INTER_LANCZOS4,
    )
    h0_resized, w0_resized, c0_resized = img0_resized.shape
    img0_resized_padded = np.zeros(
        (h0_resized + 2 * padding_pixel0, w0_resized + 2 * padding_pixel0, c0_resized),
        dtype=np.uint8,
    )
    img0_resized_padded[
        padding_pixel0 : padding_pixel0 + h0_resized,
        padding_pixel0 : padding_pixel0 + w0_resized,
    ] = np.copy(img0_resized)

    resize_factor1 = 10
    padding_pixel1 = int(900 / resize_factor1)
    h1, w1, c1 = img1.shape
    img1_resized = cv2.resize(
        img1,
        dsize=(int(w1 / resize_factor1), int(h1 / resize_factor1)),
        interpolation=cv2.INTER_LANCZOS4,
    )
    h1_resized, w1_resized, c1_resized = img1_resized.shape
    img1_resized_padded = np.zeros(
        (h1_resized + 2 * padding_pixel1, w1_resized + 2 * padding_pixel1, c1_resized),
        dtype=np.uint8,
    )
    img1_resized_padded[
        padding_pixel1 : padding_pixel1 + h1_resized,
        padding_pixel1 : padding_pixel1 + w1_resized,
    ] = np.copy(img1_resized)

    # Run modules
    input = {
        "img0": img0_resized_padded,
        "img1": img1_resized_padded,
    }
    for module in modules:
        print("\n[{}] run({})\n".format(type(module).__name__, type(input)))
        print("\t[input] {}".format(input.keys()))
        output = module.run_module(input=input)
        print("\t[output] {}".format(output.keys()))
        input = output

    print("\t[rotation_matrix] {}".format(output["rotation_matrix"]))
    print("\t[translation_matrix_zero] {}".format(output["translation_matrix_zero"]))
    print(
        "\t[translation_matrix_center] {}".format(output["translation_matrix_center"])
    )

    img1_concat_rotated = TempUtils.make_concat_rotated_images(
        img1_resized_padded,
        output["rotation_matrix"],
        output["translation_matrix_center"],
    )

    img1_rotated = TempUtils.make_rotated_image(
        img1_resized_padded,
        output["rotation_matrix"],
        output["translation_matrix_center"],
    )

    img0_resized_padded_rgb = cv2.cvtColor(img0_resized_padded, cv2.COLOR_BGR2RGB)
    img0_resized_padded_rgb_float = img0_resized_padded_rgb.astype("float32") / 255
    img01_concat = TempUtils.concat_images_different_size(
        img0_resized_padded_rgb_float, img1_concat_rotated
    )

    winname = "out"
    for name in names:
        winname = winname + "-" + str(name)

    print(
        "img0_resized_padded_rgb.shape", img0_resized_padded_rgb.shape
    )  # (506, 586, 3)
    print(
        "img0_resized_padded_rgb_float.shape", img0_resized_padded_rgb_float.shape
    )  # (506, 586, 3)

    print("img0_resized_padded.shape", img0_resized_padded.shape)  # (506, 586, 3)
    print("img1_resized_padded.shape", img1_resized_padded.shape)  # (545, 667, 3)
    print("img1_rotated.shape", img1_rotated.shape)  # (545, 667, 3)

    h0, w0, c0 = img0_resized_padded.shape
    h1, w1, c1 = img1_resized_padded.shape
    h_max, w_max = max(h0, h1), max(w0, w1)

    img0_max_canvas = np.zeros((h_max, w_max, c0), dtype=np.uint8)  # (545, 667, 3)

    img0_max_canvas[
        int((h_max - h0) / 2) : int(((h_max - h0) / 2) + h0),
        int((w_max - w0) / 2) : int(((w_max - w0) / 2) + w0),
    ] = np.copy(img0_resized_padded)

    print("img0_max_canvas.shape", img0_max_canvas.shape)

    cv2.imshow("Original", img0_max_canvas)
    cv2.imshow("Target", img1_resized_padded)
    cv2.imshow(winname, img1_rotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # PlotUtils.show_image("img01_concat", img01_concat) ###
    # PlotUtils.show_image(winname, img1_rotated)


if __name__ == "__main__":
    config = TempUtils.load_config("config/test_pipeline.yaml")
    main(config)
