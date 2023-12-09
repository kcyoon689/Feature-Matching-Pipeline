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


original = cv2.imread("/home/yoonk/Desktop/pipline/result-score/00_Original.png")
target = cv2.imread("/home/yoonk/Desktop/pipline/result-score/01_Target.png")
BRIEFBF = cv2.imread("/home/yoonk/Desktop/pipline/result-score/out-BRIEF-BFMatcher.png")
BRIEFFLANN = cv2.imread(
    "/home/yoonk/Desktop/pipline/result-score/out-BRIEF-FLANNMatcher.png"
)
LoFTR = cv2.imread("/home/yoonk/Desktop/pipline/result-score/out-LoFTR.png")
ORBBF = cv2.imread("/home/yoonk/Desktop/pipline/result-score/out-ORB-BFMatcher.png")
ORBFLANN = cv2.imread(
    "/home/yoonk/Desktop/pipline/result-score/out-ORB-FLANNMatcher.png"
)
SIFT = cv2.imread("/home/yoonk/Desktop/pipline/result-score/out-SIFT-BFMatcher.png")

gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
gray_BRIEFBF = cv2.cvtColor(BRIEFBF, cv2.COLOR_BGR2GRAY)
gray_BRIEFFLANN = cv2.cvtColor(BRIEFFLANN, cv2.COLOR_BGR2GRAY)
gray_LoFTR = cv2.cvtColor(LoFTR, cv2.COLOR_BGR2GRAY)
gray_ORBBF = cv2.cvtColor(ORBBF, cv2.COLOR_BGR2GRAY)
gray_ORBFLANN = cv2.cvtColor(ORBFLANN, cv2.COLOR_BGR2GRAY)
gray_SIFT = cv2.cvtColor(SIFT, cv2.COLOR_BGR2GRAY)

(score1, diff1) = ssim(gray_original, gray_target, full=True)
(score2, diff2) = ssim(gray_original, gray_BRIEFBF, full=True)
(score3, diff3) = ssim(gray_original, gray_BRIEFFLANN, full=True)
(score4, diff4) = ssim(gray_original, gray_LoFTR, full=True)
(score5, diff5) = ssim(gray_original, gray_ORBBF, full=True)
(score6, diff6) = ssim(gray_original, gray_ORBFLANN, full=True)
(score7, diff7) = ssim(gray_original, gray_SIFT, full=True)

original_target = (diff1 * 255).astype("uint8")
original_BRIEFBF = (diff2 * 255).astype("uint8")
original_BRIEFFLANN = (diff3 * 255).astype("uint8")
original_LoFTR = (diff4 * 255).astype("uint8")
original_ORBBF = (diff5 * 255).astype("uint8")
original_ORBFLANN = (diff6 * 255).astype("uint8")
original_SIFT = (diff7 * 255).astype("uint8")

# cv2.imshow("diff-original_target", diff1)
# cv2.imshow("diff-original_BRIEFBF", diff2)
# cv2.imshow("diff-original_BRIEFFLANN", diff3)
# cv2.imshow("diff4-original_LoFTR", diff4)
# cv2.imshow("diff5-original_ORBBF", diff5)
# cv2.imshow("diff6-original_ORBFLANN", diff6)
# cv2.imshow("diff7-original_SIFT", diff7)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
