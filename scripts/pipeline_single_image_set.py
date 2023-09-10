import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # nopep8
from feature_modules import LoFTR, ORB, SIFT, SVDTF
from utils import TempUtils


def main(config: dict):
    modules = []
    if config['SIFT'] is True:
        modules.append(SIFT())

    if config['ORB'] is True:
        modules.append(ORB())

    if config['LoFTR'] is True:
        modules.append(LoFTR())

    if config['SVDTF'] is True:
        modules.append(SVDTF())

    img0 = cv2.imread('./images/oxford.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./images/oxford2.jpg', cv2.IMREAD_COLOR)

    input = {
        'img0': img0,
        'img1': img1,
    }
    for module in modules:
        print("\n[{}] run({})\n".format(type(module).__name__, type(input)))
        print("\t[input] ".format(input.keys()))
        output = module.run(input=input)
        print("\t[output] ".format(output.keys()))
        input = output

    print(output['rotation_matrix'])
    print(output['translation_matrix_zero'])
    print(output['translation_matrix_center'])


if __name__ == '__main__':
    config = TempUtils.load_config('config/test_pipeline.yaml')
    main(config)
