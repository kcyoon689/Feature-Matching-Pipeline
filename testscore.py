import cv2
from skimage import io
from skimage.metrics import structural_similarity as ssim
import argparse
import imutils
from math import log10, sqrt
import numpy as np
import torch
from IQA_pytorch import SSIM, utils
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


base_path = "/home/yoonk/Desktop/pipline/result-score/"

method_list = [
    "original",
    "target",
    "BRIEFBF",
    "BRIEFFLANN",
    "LoFTR",
    "ORBBF",
    "ORBFLANN",
    "SIFT",
]
img_file_list = [
    "00_Original.png",
    "01_Target.png",
    "out-BRIEF-BFMatcher.png",
    "out-BRIEF-FLANNMatcher.png",
    "out-LoFTR.png",
    "out-ORB-BFMatcher.png",
    "out-ORB-FLANNMatcher.png",
    "out-SIFT-BFMatcher.png",
]
dict_list = ["rgb_img", "gray_img", "cuda_img"]

img_dict = {}

for dict_name in dict_list:
    img_dict[dict_name] = {}

for method, img_file in zip(method_list, img_file_list):
    img_dict["rgb_img"][method] = cv2.imread(base_path + img_file)
    img_dict["gray_img"][method] = convert_to_gray(img_dict["rgb_img"][method])
    img_dict["cuda_img"][method] = utils.prepare_image(
        Image.open(base_path + img_file).convert("RGB")
    ).to(device)

rgb_name_list = list(img_dict["rgb_img"].keys())
gray_name_list = list(img_dict["gray_img"].keys())
cuda_name_list = list(img_dict["cuda_img"].keys())

rgb_name_list.remove("original")
gray_name_list.remove("original")
cuda_name_list.remove("original")

score_list = [
    "original_target",
    "original_BRIEFBF",
    "original_BRIEFFLANN",
    "original_LoFTR",
    "original_ORBBF",
    "original_ORBFLANN",
    "original_SIFT",
]

SSIM_score = [0] * len(score_list)
SSIM_diff = [0] * len(score_list)
SSIM_score2 = []
PSNR_list = []


model = SSIM(channels=3)

method_list = list(img_dict["gray_img"].keys())
method_list.remove("original")
for idx, method in enumerate(method_list):
    print(len(SSIM_score))
    print(len(SSIM_diff))
    print("idx", idx)

    SSIM_score[idx], SSIM_diff[idx] = ssim(
        img_dict["gray_img"]["original"], img_dict["gray_img"][method], full=True
    )
    SSIM_diff[idx] = (SSIM_diff[idx] * 255).astype("uint8")

    score = model(
        img_dict["cuda_img"]["original"], img_dict["cuda_img"][method], as_loss=False
    )
    SSIM_score2.append(score.item())

for PSNR_idx in range(len(rgb_name_list)):
    PSNR_score = PSNR(img_dict["rgb_img"]["original"], img_dict["rgb_img"][rgb_name_list[PSNR_idx]])
    PSNR_list.append(PSNR_score)

SSIM_score_dict = {"diff_img": {}, "SSIM_score1": {}, "SSIM_score2": {}}

PSNR_score_dict = {}

for idx, score in enumerate(score_list):
    SSIM_score_dict["diff_img"][score] = SSIM_diff[idx]
    SSIM_score_dict["SSIM_score1"][score] = SSIM_score[idx]
    SSIM_score_dict["SSIM_score2"][score] = SSIM_score2[idx]

for idx, score in enumerate(score_list):
    PSNR_score_dict[score] = PSNR_list[idx]

print("SSIM_score1")
print(SSIM_score_dict["SSIM_score1"])
print("SSIM_score2")
print(SSIM_score_dict["SSIM_score2"])
print("PSNR_score(db)")
print(PSNR_score_dict)

# # # cv2.imshow("diff-original_target", diff1)
# # # cv2.imshow("diff-original_BRIEFBF", diff2)
# # # cv2.imshow("diff-original_BRIEFFLANN", diff3)
# # # cv2.imshow("diff4-original_LoFTR", diff4)
# # # cv2.imshow("diff5-original_ORBBF", diff5)
# # # cv2.imshow("diff6-original_ORBFLANN", diff6)
# # # cv2.imshow("diff7-original_SIFT", diff7)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
