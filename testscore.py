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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

base_path = "/home/yoonk/Desktop/pipline/result-score/"
method_list = ["original", "target", "BRIEFBF", "BRIEFFLANN", "LoFTR", "ORBBF", "ORBFLANN", "SIFT"]

img_dict = {
    "rgb_img":
    {
        method_list[0]: cv2.imread(base_path + "00_Original.png"),
        method_list[1]: cv2.imread(base_path + "01_Target.png"),
        method_list[2]: cv2.imread(base_path + "out-BRIEF-BFMatcher.png"),
        method_list[3]: cv2.imread(base_path + "out-BRIEF-FLANNMatcher.png"),
        method_list[4]: cv2.imread(base_path + "out-LoFTR.png"),
        method_list[5]: cv2.imread(base_path + "out-ORB-BFMatcher.png"),
        method_list[6]: cv2.imread(base_path + "out-ORB-FLANNMatcher.png"),
        method_list[7]: cv2.imread(base_path + "out-SIFT-BFMatcher.png")
    },
    "gray_img":
    {
        method_list[0]: convert_to_gray(cv2.imread(base_path + "00_Original.png")),
        method_list[1]: convert_to_gray(cv2.imread(base_path + "01_Target.png")),
        method_list[2]: convert_to_gray(cv2.imread(base_path + "out-BRIEF-BFMatcher.png")),
        method_list[3]: convert_to_gray(cv2.imread(base_path + "out-BRIEF-FLANNMatcher.png")),
        method_list[4]: convert_to_gray(cv2.imread(base_path + "out-LoFTR.png")),
        method_list[5]: convert_to_gray(cv2.imread(base_path + "out-ORB-BFMatcher.png")),
        method_list[6]: convert_to_gray(cv2.imread(base_path + "out-ORB-FLANNMatcher.png")),
        method_list[7]: convert_to_gray(cv2.imread(base_path + "out-SIFT-BFMatcher.png"))
    }
}

img_dict2 = {
    "original": utils.prepare_image(Image.open(base_path + "00_Original.png").convert("RGB")).to(device),
    "target": utils.prepare_image(Image.open(base_path + "01_Target.png").convert("RGB")).to(device),
    "BRIEFBF": utils.prepare_image(Image.open(base_path + "out-BRIEF-BFMatcher.png").convert("RGB")).to(device),
    "BRIEFFLANN": utils.prepare_image(Image.open(base_path + "out-BRIEF-FLANNMatcher.png").convert("RGB")).to(device),
    "LoFTR": utils.prepare_image(Image.open(base_path + "out-LoFTR.png").convert("RGB")).to(device),
    "ORBBF": utils.prepare_image(Image.open(base_path + "out-ORB-BFMatcher.png").convert("RGB")).to(device),
    "ORBFLANN": utils.prepare_image(Image.open(base_path + "out-ORB-FLANNMatcher.png").convert("RGB")).to(device),
    "SIFT": utils.prepare_image(Image.open(base_path + "out-SIFT-BFMatcher.png").convert("RGB")).to(device)
}

rgb_list = list(img_dict["rgb_img"].keys())
gray_list = list(img_dict["gray_img"].keys())
img2_list = list(img_dict2.keys())

rgb_list.remove("original")
gray_list.remove("original")
img2_list.remove("original")

score_list = ["original_target", "original_BRIEFBF", "original_BRIEFFLANN", "original_LoFTR", "original_ORBBF", "original_ORBFLANN", "original_SIFT"]

SSIM_score = [0] * len(score_list)
SSIM_diff = [0] * len(score_list)
SSIM_score2 = []
PSNR_list = []

model = SSIM(channels=3)

for SSIM_idx in range(len(gray_list)):
    (SSIM_score[SSIM_idx], SSIM_diff[SSIM_idx]) = ssim(img_dict["gray_img"]["original"], img_dict["gray_img"][gray_list[SSIM_idx]], full=True)
    SSIM_diff[SSIM_idx] = (SSIM_diff[SSIM_idx] * 255).astype("uint8")

for SSIM_jdx in range(len(img2_list)):
    score = (model(img_dict2["original"], img_dict2[img2_list[SSIM_jdx]], as_loss=False))
    SSIM_score2.append(score.item())

for PSNR_idx in range(len(rgb_list)):
    PSNR_score = PSNR(img_dict["rgb_img"]["original"], img_dict["rgb_img"][rgb_list[PSNR_idx]])
    PSNR_list.append(PSNR_score)

SSIM_score = {
    "Diff_img":
    {
        score_list[0]: SSIM_diff[0],
        score_list[1]: SSIM_diff[1],
        score_list[2]: SSIM_diff[2],
        score_list[3]: SSIM_diff[3],
        score_list[4]: SSIM_diff[4],
        score_list[5]: SSIM_diff[5],
        score_list[6]: SSIM_diff[6],
    },
    "SSIM_score1":
    {
        score_list[0]: SSIM_score[0],
        score_list[1]: SSIM_score[1],
        score_list[2]: SSIM_score[2],
        score_list[3]: SSIM_score[3],
        score_list[4]: SSIM_score[4],
        score_list[5]: SSIM_score[5],
        score_list[6]: SSIM_score[6],
    },
    "SSIM_score2":
    {
        score_list[0]: SSIM_score2[0],
        score_list[1]: SSIM_score2[1],
        score_list[2]: SSIM_score2[2],
        score_list[3]: SSIM_score2[3],
        score_list[4]: SSIM_score2[4],
        score_list[5]: SSIM_score2[5],
        score_list[6]: SSIM_score2[6],
    },
}

PSNR_score = {
    score_list[0]: PSNR_list[0],
    score_list[1]: PSNR_list[1],
    score_list[2]: PSNR_list[2],
    score_list[3]: PSNR_list[3],
    score_list[4]: PSNR_list[4],
    score_list[5]: PSNR_list[5],
    score_list[6]: PSNR_list[6],
}

print("SSIM_score1")
print(SSIM_score["SSIM_score1"])
print("SSIM_score2")
print(SSIM_score["SSIM_score2"])
print("PSNR_score(db)")
print(PSNR_score)

# # # cv2.imshow("diff-original_target", diff1)
# # # cv2.imshow("diff-original_BRIEFBF", diff2)
# # # cv2.imshow("diff-original_BRIEFFLANN", diff3)
# # # cv2.imshow("diff4-original_LoFTR", diff4)
# # # cv2.imshow("diff5-original_ORBBF", diff5)
# # # cv2.imshow("diff6-original_ORBFLANN", diff6)
# # # cv2.imshow("diff7-original_SIFT", diff7)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()
