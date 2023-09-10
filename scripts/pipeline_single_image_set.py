import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8
from feature_modules import LoFTR, ORB, SIFT, SVDTF
from utils import TempUtils


def main(config: dict):
    # Load modules
    modules = []
    if config['SIFT'] is True:
        modules.append(SIFT())

    if config['ORB'] is True:
        modules.append(ORB())

    if config['LoFTR'] is True:
        modules.append(LoFTR())

    if config['SVDTF'] is True:
        modules.append(SVDTF())

    # Image preprocessing
    img0 = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./images/oxford2.jpg', cv2.IMREAD_COLOR)

    resize_factor = 2
    h0, w0, c0 = img0.shape
    img0_resized = cv2.resize(img0, dsize=(
        int(w0/resize_factor), int(h0/resize_factor)), interpolation=cv2.INTER_LANCZOS4)

    resize_factor = 4
    h1, w1, c1 = img1.shape
    img1_resized = cv2.resize(img1, dsize=(
        int(w1/resize_factor), int(h1/resize_factor)), interpolation=cv2.INTER_LANCZOS4)

    # Run modules
    input = {
        'img0': img0_resized,
        'img1': img1_resized,
    }
    for module in modules:
        print("\n[{}] run({})\n".format(type(module).__name__, type(input)))
        print("\t[input] {}".format(input.keys()))
        output = module.run_module(input=input)
        print("\t[output] {}".format(output.keys()))
        input = output

    print("\t[rotation_matrix] {}".format(output['rotation_matrix']))
    print("\t[translation_matrix_zero] {}".format(
        output['translation_matrix_zero']))
    print("\t[translation_matrix_center] {}".format(
        output['translation_matrix_center']))


if __name__ == '__main__':
    config = TempUtils.load_config('config/test_pipeline.yaml')
    main(config)
