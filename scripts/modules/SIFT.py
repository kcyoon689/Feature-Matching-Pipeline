# SIFT로 특징점 및 디스크립터 추출(desc_sift.py)

import cv2

class SIFT:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()

    def run(self, img_rgb, image_output=False):
        # SIFT 추출기 생성
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        keypoints, descriptor = self.sift.detectAndCompute(img_gray, None)

        for i in keypoints:
            x, y = i.pt
            # print("x",x)
            # print("y",y)

        # 키 포인트 그리기
        img_draw = cv2.drawKeypoints(img_rgb, keypoints, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if image_output is True:
            return keypoints, descriptor, img_draw
        else:
            return keypoints, descriptor


if __name__ == "__main__":
    img = cv2.imread('./images/oxford.jpg')

    sift = SIFT()
    _, _, img_output = sift.run(img, image_output=True)

    cv2.imshow('sift', img_output)
    cv2.waitKey()
    cv2.destroyAllWindows()
