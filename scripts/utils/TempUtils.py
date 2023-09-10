import numpy as np
from PIL import Image as PImage
import plotly.graph_objects as go
import yaml


class TempUtils:
    def load_config(file_path: str) -> dict:
        with open(file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print("config: {}\n".format(config))
            return config

    def get_transformation_from_rotation(offset_rotation_deg):
        transformation = np.eye(3, dtype=np.float32)
        transformation[0][0] = np.cos(offset_rotation_deg * np.pi / 180.0)
        transformation[0][1] = -np.sin(offset_rotation_deg * np.pi / 180.0)
        transformation[1][0] = np.sin(offset_rotation_deg * np.pi / 180.0)
        transformation[1][1] = np.cos(offset_rotation_deg * np.pi / 180.0)

        return transformation

    def apply_transformation_to_image_feature(image_feature, transformation, axis_order):
        image_feature[axis_order[0]] = image_feature[1]
        image_feature[axis_order[1]] = image_feature[0]
        image_feature[axis_order[2]] = [0] * len(image_feature)
        image_feature[axis_order[0] + '_r'] = transformation[0][0] * \
            image_feature[axis_order[0]] + \
            transformation[0][1] * image_feature[axis_order[1]]
        image_feature[axis_order[1] + '_r'] = transformation[1][0] * \
            image_feature[axis_order[0]] + \
            transformation[1][1] * image_feature[axis_order[1]]
        image_feature[axis_order[0] +
                      '_tf'] = image_feature[axis_order[0] + '_r'] + transformation[0][2]
        image_feature[axis_order[1] +
                      '_tf'] = image_feature[axis_order[1] + '_r'] + transformation[1][2]

        return image_feature

    def apply_offset_to_image_feature(image_feature, plane_offset, axis_order, idx):
        image_feature[axis_order[0]] += plane_offset[axis_order[0]][idx]
        image_feature[axis_order[1]] += plane_offset[axis_order[1]][idx]
        image_feature[axis_order[2]] += plane_offset[axis_order[2]][idx]
        image_feature[axis_order[0] + '_r'] += plane_offset[axis_order[0]][idx]
        image_feature[axis_order[1] + '_r'] += plane_offset[axis_order[1]][idx]
        image_feature[axis_order[0] +
                      '_tf'] += plane_offset[axis_order[0]][idx]
        image_feature[axis_order[1] +
                      '_tf'] += plane_offset[axis_order[1]][idx]

        return image_feature

    # remove unused argument

    def create_xy_rgb_surface(cv_img_rgb, z, x_offset=0, y_offset=0, black_criteria=50, **kwargs):
        cv_img_rgb = (cv_img_rgb * 255).astype(np.uint8)
        cv_img_rgb = cv_img_rgb.swapaxes(0, 1)[:, ::-1]
        eight_bit_img = PImage.fromarray(cv_img_rgb, mode='RGB').convert(
            'P', palette='WEB', dither=None)
        x, y = np.linspace(x_offset, x_offset + eight_bit_img.width, eight_bit_img.width), np.linspace(
            y_offset, y_offset + eight_bit_img.height, eight_bit_img.height)
        idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
        colorscale = [[i/255.0, 'rgb({}, {}, {})'.format(*rgb)]
                      for i, rgb in enumerate(idx_to_color)]

        z_array = np.ones_like(eight_bit_img) * z
        eight_bit_img_np = np.array(eight_bit_img)

        z_filtered = np.zeros_like(z_array, dtype=np.float32)
        for x_idx in range(len(x)):
            for y_idx in range(len(y)):
                if eight_bit_img_np[y_idx][x_idx] < black_criteria:
                    z_filtered[y_idx][x_idx] = np.nan
                else:
                    z_filtered[y_idx][x_idx] = z_array[y_idx][x_idx]

        return go.Surface(
            x=x,
            y=y,
            z=z_filtered,
            surfacecolor=eight_bit_img_np,
            cmin=0,
            cmax=255,
            colorscale=colorscale,
            showscale=False,
            **kwargs
        )

    # remove unused argument

    def create_yz_rgb_surface(cv_img_rgb, x, y_offset=0, z_offset=0, black_criteria=50, **kwargs):
        cv_img_rgb = (cv_img_rgb * 255).astype(np.uint8)
        cv_img_rgb = cv_img_rgb.swapaxes(0, 1)[:, ::-1]
        eight_bit_img = PImage.fromarray(cv_img_rgb, mode='RGB').convert(
            'P', palette='WEB', dither=None)
        z, y = np.linspace(z_offset, z_offset + eight_bit_img.width, eight_bit_img.width), np.linspace(
            y_offset, y_offset + eight_bit_img.height, eight_bit_img.height)
        idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
        colorscale = [[i/255.0, 'rgb({}, {}, {})'.format(*rgb)]
                      for i, rgb in enumerate(idx_to_color)]

        x_array = np.ones(len(z)) * x  # 507
        z_array = np.array([z] * len(y))  # 677, 507
        eight_bit_img_np = np.array(eight_bit_img)  # 677, 507

        z_filtered = np.zeros_like(z_array, dtype=np.float32)  # 677, 507
        for y_idx in range(len(y)):
            for z_idx in range(len(z)):
                if eight_bit_img_np[y_idx][z_idx] < black_criteria:
                    z_filtered[y_idx][z_idx] = np.nan
                else:
                    z_filtered[y_idx][z_idx] = z_array[y_idx][z_idx]

        return go.Surface(
            x=x_array,  # 507
            y=y,  # 677
            z=z_filtered,  # 677, 507
            surfacecolor=eight_bit_img_np,
            cmin=0,
            cmax=255,
            colorscale=colorscale,
            showscale=False,
            **kwargs
        )

    def get_plot_data_image(cv_image_list, plane_offset={'x': [], 'y': [], 'z': []}, align='h'):
        plot_data_list = []

        for cv_image, x_offset, y_offset, z_offset in zip(cv_image_list, plane_offset['x'], plane_offset['y'], plane_offset['z']):
            if align == 'h':
                plot_data_list.append(
                    create_xy_rgb_surface(
                        cv_img_rgb=np.flipud(cv_image),
                        z=z_offset,
                        x_offset=x_offset,
                        y_offset=y_offset,
                        contours_z=dict(show=True, project_z=True,
                                        highlightcolor='limegreen'),
                        opacity=1.0
                    )
                )
            elif align == 'v':
                plot_data_list.append(
                    create_yz_rgb_surface(
                        cv_img_rgb=np.flipud(cv_image),
                        x=x_offset,
                        y_offset=y_offset,
                        z_offset=z_offset,
                        contours_x=dict(show=True, project_x=True,
                                        highlightcolor='limegreen'),
                        opacity=1.0
                    )
                )

        return plot_data_list

    def get_plot_data_feature_points(feature_df_list, feature_color_list, feature_point_size, axis_order, is_apply_tf=False, is_plot_line=False):
        plot_data_list = []
        axis_x = 'x_tf' if is_apply_tf is True and axis_order[2] != 'x' else 'x'
        axis_y = 'y_tf' if is_apply_tf is True and axis_order[2] != 'y' else 'y'
        axis_z = 'z_tf' if is_apply_tf is True and axis_order[2] != 'z' else 'z'

        for idx in range(0, int(len(feature_df_list) / 2)):
            plot_data_list.append(go.Scatter3d(
                x=feature_df_list[2*idx][axis_x],
                y=feature_df_list[2*idx][axis_y],
                z=feature_df_list[2*idx][axis_z],
                mode='markers',
                marker=dict(
                    color=feature_color_list[idx],
                    size=feature_point_size,
                ),
            ))
            plot_data_list.append(go.Scatter3d(
                x=feature_df_list[2*idx+1][axis_x],
                y=feature_df_list[2*idx+1][axis_y],
                z=feature_df_list[2*idx+1][axis_z],
                mode='markers',
                marker=dict(
                    color=feature_color_list[idx],
                    size=feature_point_size,
                ),
            ))

            if is_plot_line:
                for line_x1, line_x2, line_y1, line_y2, line_z1, line_z2 in zip(
                        feature_df_list[2 *
                                        idx][axis_x], feature_df_list[2*idx+1][axis_x],
                        feature_df_list[2 *
                                        idx][axis_y], feature_df_list[2*idx+1][axis_y],
                        feature_df_list[2*idx][axis_z], feature_df_list[2*idx+1][axis_z]):
                    plot_data_list.append(go.Scatter3d(
                        x=[line_x1, line_x2],
                        y=[line_y1, line_y2],
                        z=[line_z1, line_z2],
                        mode='lines',
                        marker=dict(
                            color=feature_color_list[idx],
                        ),
                    ))

        return plot_data_list

    def get_plot_data_tf_verification(feature_df0, feature_df1, feature_color_list, axis_order, index):
        plot_data_list = []

        # mean points
        feature_df0_mean = {
            axis_order[0]: np.mean(feature_df0[axis_order[0]]),
            axis_order[1]: np.mean(feature_df0[axis_order[1]]),
            axis_order[2]: np.mean(feature_df0[axis_order[2]]),
            axis_order[0] + '_r': np.mean(feature_df0[axis_order[0] + '_r']),
            axis_order[1] + '_r': np.mean(feature_df0[axis_order[1] + '_r']),
            axis_order[0] + '_tf': np.mean(feature_df0[axis_order[0] + '_tf']),
            axis_order[1] + '_tf': np.mean(feature_df0[axis_order[1] + '_tf']),
        }
        feature_df1_mean = {
            axis_order[0] + '_tf': np.mean(feature_df1[axis_order[0] + '_tf']),
            axis_order[1] + '_tf': np.mean(feature_df1[axis_order[1] + '_tf']),
        }

        # feature points
        plot_data_list.append(go.Scatter3d(x=feature_df0[axis_order[0]],
                                           y=feature_df0[axis_order[1]],
                                           z=feature_df0[axis_order[2]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index],
            size=3,
        ),
        ))
        plot_data_list.append(go.Scatter3d(x=feature_df0[axis_order[0] + '_r'],
                                           y=feature_df0[axis_order[1] + '_r'],
                                           z=feature_df0[axis_order[2]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index+1],
            size=3,
        ),
        ))
        plot_data_list.append(go.Scatter3d(x=feature_df0[axis_order[0] + '_tf'],
                                           y=feature_df0[axis_order[1] + '_tf'],
                                           z=feature_df0[axis_order[2]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index+2],
            size=3,
        ),
        ))
        plot_data_list.append(go.Scatter3d(x=feature_df1[axis_order[0] + '_tf'],
                                           y=feature_df1[axis_order[1] + '_tf'],
                                           z=feature_df0[axis_order[2]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index+3],
            size=3,
        ),
        ))

        for line_x0, line_x0_r, line_y0, line_y0_r, line_z0 in zip(feature_df0[axis_order[0]], feature_df0[axis_order[0] + '_r'], feature_df0[axis_order[1]], feature_df0[axis_order[1] + '_r'], feature_df0[axis_order[2]]):
            plot_data_list.append(go.Scatter3d(x=[line_x0, line_x0_r],
                                               y=[line_y0, line_y0_r],
                                               z=[line_z0, line_z0],
                                               mode='lines',
                                               marker=dict(
                color=feature_color_list[index],
            ),
            ))

        for line_x0_r, line_x0_tf, line_y0_r, line_y0_tf, line_z0 in zip(feature_df0[axis_order[0] + '_r'], feature_df0[axis_order[0] + '_tf'], feature_df0[axis_order[1] + '_r'], feature_df0[axis_order[1] + '_tf'], feature_df0[axis_order[2]]):
            plot_data_list.append(go.Scatter3d(x=[line_x0_r, line_x0_tf],
                                               y=[line_y0_r, line_y0_tf],
                                               z=[line_z0, line_z0],
                                               mode='lines',
                                               marker=dict(
                color=feature_color_list[index+1],
            ),
            ))

        for line_x0_tf, line_x1_tf, line_y0_tf, line_y1_tf, line_z0 in zip(feature_df0[axis_order[0] + '_tf'], feature_df1[axis_order[0] + '_tf'], feature_df0[axis_order[1] + '_tf'], feature_df1[axis_order[1] + '_tf'], feature_df0[axis_order[2]]):
            plot_data_list.append(go.Scatter3d(x=[line_x0_tf, line_x1_tf],
                                               y=[line_y0_tf, line_y1_tf],
                                               z=[line_z0, line_z0],
                                               mode='lines',
                                               marker=dict(
                color=feature_color_list[index+2],
            ),
            ))

        # mean points
        plot_data_list.append(go.Scatter3d(x=[feature_df0_mean[axis_order[0]]],
                                           y=[feature_df0_mean[axis_order[1]]],
                                           z=[feature_df0_mean[axis_order[2]]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index],
            size=9,
        ),
        ))
        plot_data_list.append(go.Scatter3d(x=[feature_df0_mean[axis_order[0] + '_r']],
                                           y=[feature_df0_mean[axis_order[1] + '_r']],
                                           z=[feature_df0_mean[axis_order[2]]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index+1],
            size=9,
        ),
        ))
        plot_data_list.append(go.Scatter3d(x=[feature_df0_mean[axis_order[0] + '_tf']],
                                           y=[feature_df0_mean[axis_order[1] + '_tf']],
                                           z=[feature_df0_mean[axis_order[2]]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index+2],
            size=9,
        ),
        ))
        plot_data_list.append(go.Scatter3d(x=[feature_df1_mean[axis_order[0] + '_tf']],
                                           y=[feature_df1_mean[axis_order[1] + '_tf']],
                                           z=[feature_df0_mean[axis_order[2]]],
                                           mode='markers',
                                           marker=dict(
            color=feature_color_list[index+3],
            size=9,
        ),
        ))

        plot_data_list.append(go.Scatter3d(x=[feature_df0_mean[axis_order[0]], feature_df0_mean[axis_order[0] + '_r']],
                                           y=[feature_df0_mean[axis_order[1]],
                                              feature_df0_mean[axis_order[1] + '_r']],
                                           z=[feature_df0_mean[axis_order[2]],
                                              feature_df0_mean[axis_order[2]]],
                                           mode='lines',
                                           marker=dict(
            color=feature_color_list[index],
        ),
        ))

        plot_data_list.append(go.Scatter3d(x=[feature_df0_mean[axis_order[0] + '_r'], feature_df0_mean[axis_order[0] + '_tf']],
                                           y=[feature_df0_mean[axis_order[1] + '_r'],
                                              feature_df0_mean[axis_order[1] + '_tf']],
                                           z=[feature_df0_mean[axis_order[2]],
                                              feature_df0_mean[axis_order[2]]],
                                           mode='lines',
                                           marker=dict(
            color=feature_color_list[index+1],
        ),
        ))

        return plot_data_list
