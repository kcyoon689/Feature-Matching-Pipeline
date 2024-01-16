from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import argparse

torch.set_grad_enabled(False)
# images = Path("assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

parser = argparse.ArgumentParser(description='LightGlue Method')
parser.add_argument('--img1', type=str, default='/home/yoonk/Desktop/test/images/E05_resize_10.png', help='path to image 1')
parser.add_argument('--img2', type=str, default='/home/yoonk/Desktop/test/images/E07_resize_10.png', help='path to image 2')
parser.add_argument('--method', type=str, default='superpoint', help='method to use')

args = parser.parse_args()

extractor = args.method

if extractor == 'superpoint':
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

if extractor == 'sift':
    extractor = SIFT(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="sift").eval().to(device)

if extractor == 'aliked':
    extractor = ALIKED(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="aliked").eval().to(device)

if extractor == 'disk':
    extractor = DISK(max_num_keypoints=2048).eval().to(device)  # load the extractor
    matcher = LightGlue(features="disk").eval().to(device)

image0 = load_image(args.img1)
image1 = load_image(args.img2)

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

# axes = viz2d.plot_images([image0, image1])
# viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
# viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

# kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
# viz2d.plot_images([image0, image1])
# viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)

