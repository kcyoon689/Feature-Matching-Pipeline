import pandas as pd
from typing import Tuple


class DataFrameUtils:
    def drop_zero(df: pd.DataFrame) -> pd.DataFrame:
        idx = df[df[0] == 0].index
        df.drop(idx, inplace=True)
        return df

    def make_data_frame_from_keypoints(keypoints: Tuple) -> pd.DataFrame:
        keypoints_x = []
        keypoints_y = []

        for keypoint in keypoints:
            x, y = keypoint.pt
            keypoints_x.append(x)
            keypoints_y.append(y)

        keypoints_df = pd.DataFrame(
            {
                "keypoints": keypoints,
                "x": keypoints_x,
                "y": keypoints_y,
            }
        )

        return keypoints_df
