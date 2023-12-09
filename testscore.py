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

def Togray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

base_path = "/home/yoonk/Desktop/pipline/result-score/"
img_list = []
gray_list = []
PSNR_list = []

img_dict = {
    "original": cv2.imread(base_path + "00_Original.png"),
    "target": cv2.imread(base_path + "01_Target.png"),
    "BRIEFBF": cv2.imread(base_path + "out-BRIEF-BFMatcher.png"),
    "BRIEFFLANN": cv2.imread(base_path + "out-BRIEF-FLANNMatcher.png"),
    "LoFTR": cv2.imread(base_path + "out-LoFTR.png"),
    "ORBBF": cv2.imread(base_path + "out-ORB-BFMatcher.png"),
    "ORBFLANN": cv2.imread(base_path + "out-ORB-FLANNMatcher.png"),
    "SIFT": cv2.imread(base_path + "out-SIFT-BFMatcher.png")
}

img_list = list(img_dict.keys())
img_list.remove("original")

gray_original = Togray(img_dict["original"])
gray_target = Togray(img_dict["target"])
gray_BRIEFBF = Togray(img_dict["BRIEFBF"])
gray_BRIEFFLANN = Togray(img_dict["BRIEFFLANN"])
gray_LoFTR = Togray(img_dict["LoFTR"])
gray_ORBBF = Togray(img_dict["ORBBF"])
gray_ORBFLANN = Togray(img_dict["ORBFLANN"])
gray_SIFT = Togray(img_dict["SIFT"])

gray_list = [gray_target, gray_BRIEFBF, gray_BRIEFFLANN, gray_LoFTR, gray_ORBBF, gray_ORBFLANN, gray_SIFT]
SSIM_score = [0, 0, 0, 0, 0, 0, 0]
SSIM_diff = [0, 0, 0, 0, 0, 0, 0]

for idx in range(len(gray_list)):
    (SSIM_score[idx], SSIM_diff[idx]) = ssim(gray_original, gray_list[idx], full=True)
    SSIM_diff[idx] = (SSIM_diff[idx] * 255).astype("uint8")

Diff = {
    "original_target": SSIM_diff[0],
    "original_BRIEFBF": SSIM_diff[1],
    "original_BRIEFFLANN": SSIM_diff[2],
    "original_LoFTR": SSIM_diff[3],
    "original_ORBBF": SSIM_diff[4],
    "original_ORBFLANN": SSIM_diff[5],
    "original_SIFT": SSIM_diff[6]
}

SSIM_score1 = {
    "original_target": SSIM_score[0],
    "original_BRIEFBF": SSIM_score[1],
    "original_BRIEFFLANN": SSIM_score[2],
    "original_LoFTR": SSIM_score[3],
    "original_ORBBF": SSIM_score[4],
    "original_ORBFLANN": SSIM_score[5],
    "original_SIFT": SSIM_score[6]
}


for psnr_idx in img_list:
    PSNR_score = PSNR(img_dict["original"], img_dict[psnr_idx])
    PSNR_list.append(PSNR_score)


PSNR_score = {
    "original_target": PSNR_list[0],
    "original_BRIEFBF": PSNR_list[1],
    "original_BRIEFFLANN": PSNR_list[2],
    "original_LoFTR": PSNR_list[3],
    "original_ORBBF": PSNR_list[4],
    "original_ORBFLANN": PSNR_list[5],
    "original_SIFT": PSNR_list[6]
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

model = SSIM(channels=3)

SSIM_score_list = []

for jdx in range(len(img_list)):
    score = (model(img_dict2["original"], img_dict2[img_list[jdx]], as_loss=False))
    SSIM_score_list.append(score.item())

    
SSIM_score2_dict = {
    "original_target": SSIM_score_list[0],
    "original_BRIEFBF": SSIM_score_list[1],
    "original_BRIEFFLANN": SSIM_score_list[2],
    "original_LoFTR": SSIM_score_list[3],
    "original_ORBBF": SSIM_score_list[4],
    "original_ORBFLANN": SSIM_score_list[5],
    "original_SIFT": SSIM_score_list[6]
}

print("SSIM_score1: ")
print(SSIM_score1)

print("PSNR_score")
print(PSNR_score)

print("SSIM_score2: ")
print(SSIM_score2_dict)


# # cv2.imshow("diff-original_target", diff1)
# # cv2.imshow("diff-original_BRIEFBF", diff2)
# # cv2.imshow("diff-original_BRIEFFLANN", diff3)
# # cv2.imshow("diff4-original_LoFTR", diff4)
# # cv2.imshow("diff5-original_ORBBF", diff5)
# # cv2.imshow("diff6-original_ORBFLANN", diff6)
# # cv2.imshow("diff7-original_SIFT", diff7)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
