import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import draw_LAF_matches
import time
import pandas as pd
import imutils
import math
from scipy.linalg import svd, det
from PIL import Image as PImage
import plotly.graph_objects as go
from tqdm import tqdm, trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

def LoFTR_matchers(img0, img1):
    matcher = KF.LoFTR(pretrained='indoor_new')

    input_dict = {'image0': K.color.rgb_to_grayscale(img0), # LofTR works on grayscale images only
                'image1': K.color.rgb_to_grayscale(img1)}

    with torch.inference_mode():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.0, 0.999, 100000)
    inliers = inliers > 0

    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1, -1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1, -1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1, -1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1, -1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img0),
        K.tensor_to_image(img1),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                #    'tentative_color': (1.0, 0.5, 1),
                'tentative_color': None,
                'feature_color': (0.2, 0.5, 1), 'vertical': True})

    mkpts0_df = pd.DataFrame(mkpts0)
    mkpts1_df= pd.DataFrame(mkpts1)

    filter0 = mkpts0 * inliers
    filter1 = mkpts1 * inliers

    filter0_df = pd.DataFrame(filter0)
    filter1_df = pd.DataFrame(filter1)

    # return Fm, inliers, mkpts0_df, mkpts1_df, filter0_df, filter1_df
    return mkpts0_df, mkpts1_df, filter0_df, filter1_df
