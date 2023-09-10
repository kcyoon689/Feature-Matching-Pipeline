import cv2
import numpy as np


class PlotUtils:
    def show_image(title: str, image: np.ndarray):
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 2000, 1500)
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def plot_2d_image_with_features(img0, img1, img0_feature_df, img1_feature_df,
                                    img0_feature_matched_df, img1_feature_matched_df):
        h0, w0, c0 = img0.shape
        h1, w1, c1 = img1.shape

        # features
        for idx in range(0, len(img0_feature_df)):
            x = int(img0_feature_df[0][idx])
            y = int(img0_feature_df[1][idx])
            cv2.circle(img0, (x, y), 3, (0, 0, 255), -1)

        for idx in range(0, len(img1_feature_df)):
            x = int(img1_feature_df[0][idx])
            y = int(img1_feature_df[1][idx])
            cv2.circle(img1, (x, y), 3, (0, 0, 255), -1)

        # matched features
        for idx in range(0, len(img0_feature_matched_df)):
            x = int(img0_feature_matched_df[0][idx])
            y = int(img0_feature_matched_df[1][idx])
            cv2.circle(img0, (x, y), 3, (0, 255, 0), -1)

        for idx in range(0, len(img1_feature_matched_df)):
            x = int(img1_feature_matched_df[0][idx])
            y = int(img1_feature_matched_df[1][idx])
            cv2.circle(img1, (x, y), 3, (0, 255, 0), -1)

        img_result = np.zeros((max(h0, h1), w0+w1, c0), dtype=np.uint8)
        img_result[0:h0, 0:w0, :] = np.copy(img0)
        img_result[0:h1, w0:w0+w1, :] = np.copy(img1)

        # lines between matched features
        for idx in range(0, len(img0_feature_matched_df)):
            x0 = int(img0_feature_matched_df[0][idx])
            y0 = int(img0_feature_matched_df[1][idx])
            x1 = int(img1_feature_matched_df[0][idx])
            y1 = int(img1_feature_matched_df[1][idx])
            cv2.line(img_result, (x0, y0), (w0+x1, y1), (0, 255, 0), 1)

        return img0, img1, img_result
