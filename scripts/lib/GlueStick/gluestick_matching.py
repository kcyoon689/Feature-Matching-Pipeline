from os.path import join

import cv2
import torch
from matplotlib import pyplot as plt

from gluestick import batch_to_np, numpy_image_to_torch, GLUESTICK_ROOT
from gluestick.drawing import (
    plot_images,
    plot_lines,
    plot_color_line_matches,
    plot_keypoints,
    plot_matches,
)
from gluestick.models.two_view_pipeline import TwoViewPipeline

MAX_N_POINTS, MAX_N_LINES = 1000, 300

# Evaluation config
conf = {
    "name": "two_view_pipeline",
    "use_lines": True,
    "extractor": {
        "name": "wireframe",
        "sp_params": {
            "force_num_keypoints": False,
            "max_num_keypoints": MAX_N_POINTS,
        },
        "wireframe_params": {
            "merge_points": True,
            "merge_line_endpoints": True,
        },
        "max_n_lines": MAX_N_LINES,
    },
    "matcher": {
        "name": "gluestick",
        "weights": str(
            GLUESTICK_ROOT / "resources" / "weights" / "checkpoint_GlueStick_MD.tar"
        ),
        "trainable": False,
    },
    "ground_truth": {
        "from_pose_depth": False,
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline_model = TwoViewPipeline(conf).to(device).eval()
# pipeline_model

img_path0 = "/home/yoonk/Desktop/test/images/E05_resize_10.png"
img_path1 = "/home/yoonk/Desktop/test/images/E07_resize_10.png"

img = cv2.imread(img_path0, cv2.IMREAD_GRAYSCALE)

gray0 = cv2.imread(img_path0, 0)
gray1 = cv2.imread(img_path1, 0)

gray0 = cv2.resize(gray0, (640, 480))
gray1 = cv2.resize(gray1, (640, 480))

# Convert images into torch and execute GlueStick💥

torch_gray0, torch_gray1 = numpy_image_to_torch(gray0), numpy_image_to_torch(gray1)
torch_gray0, torch_gray1 = torch_gray0.to(device)[None], torch_gray1.to(device)[None]
x = {"image0": torch_gray0, "image1": torch_gray1}
pred = pipeline_model(x)

print(
    f"Detected Keypoints: {pred['keypoints0'].shape[1]} img1, {pred['keypoints1'].shape[1]} img2"
)
print(
    f"Matched {(pred['matches0'] >= 0).sum()} points and {(pred['line_matches0'] >= 0).sum()} lines"
)

pred = batch_to_np(pred)
kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
m0 = pred["matches0"]


valid_matches = m0 != -1
match_indices = m0[valid_matches]
matched_kps0 = kp0[valid_matches]
matched_kps1 = kp1[match_indices]
