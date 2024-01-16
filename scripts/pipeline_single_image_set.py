import os
import sys
import json
import cv2
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from feature_modules import BRIEF, ORB, SIFT
from feature_modules import BFMatcher, FLANNMatcher
from feature_modules import GlueStick, LightGlue, LoFTR
from feature_modules import SVDTF
from utils import PlotUtils, TempUtils


def run_image_registration(
    args: argparse.Namespace,
    config: dict,
    image0_path: str,
    image1_path: str,
    resize_factor0: int = 10,
    resize_factor1: int = 10,
    show_image: bool = True,
) -> dict:
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

    if config["GlueStick"] is True:
        modules.append(GlueStick())
        names.append("GlueStick")

    if config["LightGlue"] is True:
        modules.append(LightGlue(args.light_glue_method))
        names.append("LightGlue")

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
    img0 = cv2.imread(image0_path, cv2.IMREAD_COLOR)
    img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)

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

    # print("\t[rotation_matrix] {}".format(output["rotation_matrix"]))
    # print("\t[translation_matrix_zero] {}".format(output["translation_matrix_zero"]))
    # print("\t[translation_matrix_center] {}".format(output["translation_matrix_center"]))

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

    # (506, 586, 3)
    # print("img0_resized_padded_rgb.shape", img0_resized_padded_rgb.shape)
    # (506, 586, 3)
    # print("img0_resized_padded_rgb_float.shape", img0_resized_padded_rgb_float.shape)

    # print("img0_resized_padded.shape", img0_resized_padded.shape)  # (506, 586, 3)
    # print("img1_resized_padded.shape", img1_resized_padded.shape)  # (545, 667, 3)
    # print("img1_rotated.shape", img1_rotated.shape)  # (545, 667, 3)

    h0, w0, c0 = img0_resized_padded.shape
    h1, w1, c1 = img1_resized_padded.shape
    h_max, w_max = max(h0, h1), max(w0, w1)

    img0_max_canvas = np.zeros((h_max, w_max, c0), dtype=np.uint8)  # (545, 667, 3)

    img0_max_canvas[
        int((h_max - h0) / 2) : int(((h_max - h0) / 2) + h0),
        int((w_max - w0) / 2) : int(((w_max - w0) / 2) + w0),
    ] = np.copy(img0_resized_padded)

    # print("img0_max_canvas.shape", img0_max_canvas.shape)

    if show_image:
        cv2.imshow("Original", img0_max_canvas)
        cv2.imshow("Target", img1_resized_padded)
        cv2.imshow(winname, img1_rotated)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # PlotUtils.show_image("img01_concat", img01_concat) ###
    # PlotUtils.show_image(winname, img1_rotated)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Registration")
    parser.add_argument(
        "--config",
        type=str,
        default="config/test_pipeline.yaml",
        help="config file path",
    )
    parser.add_argument(
        "--gt_json",
        type=str,
        default="",  # "GT/E05_resize_10/result.json"
        help="gt json path",
    )
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
        "--light_glue_method",
        type=str,
        default="",
        help="method to use for LightGlue",
    )

    args = parser.parse_args()

    if args.gt_json != "":
        gt_result = json.load(open(args.gt_json, "r"))[0]
        image0_path = gt_result["original_image"]
        image1_path = gt_result["transformed_image"]

        print("\t[Ground Truth]")
        print("image0_path", image0_path)
        print("image1_path", image1_path)
        print("rotation_angle", gt_result["rotation_angle"], " degree")
        print("rotation_matrix", gt_result["rotation_matrix"])
        print("translation_matrix", gt_result["translation_matrix"])
    else:
        image0_path = args.img0
        image1_path = args.img1

    config = TempUtils.load_config(args.config)
    result = run_image_registration(
        config=config,
        args=args,
        image0_path=image0_path,
        image1_path=image1_path,
        resize_factor0=10,
        resize_factor1=10,
        show_image=False,
    )

    rotation_angle = np.arctan2(
        result["rotation_matrix"][0][1], result["rotation_matrix"][0][0]
    )
    rotation_angle *= 180 / np.pi

    print("\t[Result]")
    print("rotation_angle", rotation_angle, " degree")
    print("rotation_matrix", result["rotation_matrix"])
    print("translation_matrix_zero", result["translation_matrix_zero"])

    if args.gt_json != "":
        print("\t[Difference]")
        print("rotation_angle_diff", rotation_angle - gt_result["rotation_angle"])
        translation_matrix_diff = [
            gt_result["translation_matrix"][0] - result["translation_matrix_zero"][1],
            gt_result["translation_matrix"][1] - result["translation_matrix_zero"][0],
        ]
        print("translation_matrix_diff", translation_matrix_diff)
        print(
            "translation_matrix_diff_scale",
            translation_matrix_diff[0] / gt_result["width"],
            translation_matrix_diff[1] / gt_result["height"],
            np.sqrt(translation_matrix_diff[0] ** 2 + translation_matrix_diff[1] ** 2)
            / np.sqrt(gt_result["width"] ** 2 + gt_result["height"] ** 2),
            "%",
        )
