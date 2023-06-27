# --*-- UTF-8- --*--
# Data  :  上午9:59
# Name  : generate_augmented_pc.py

from pathlib import Path
import numpy as np
import math
import random
import open3d as o3d
import laspy
import argparse

import yaml
from sklearn.cluster import DBSCAN

from helper_ply import write_ply, read_ply

def generate_augmented_tiles(tilecodes, args, translation=False, scaling=False, cloud_rotation=False,
                             add_normals=True, manual_check=True, remap=False):
    class_names = args.class_names
    class_label = args.class_label  # label of object of interest to crop tile for.
    class_name = class_names[class_label]
    tile_radius = args.tile_radius  # radius of tile.
    translation_factor_limits = args.translation_factor_limits
    scaling_factor_limits = args.scaling_factor_limits  # world scaling factor to scale complete tile.
    degree_limit_tile_rotation = args.degree_limit_tile_rotation # maximum rotation degree of complete tile.
    reproduce_percentage = args.reproduce_percentage  # what probability to make new tile out of object of interest. range 0-1.
    augm_type = args.augm_type
    total = 0

    for i, in_file in enumerate(tilecodes):

        tilecode = in_file.stem

        # ************* load default point cloud ***************
        if args.data_format == 'las':
            laz_file = laspy.read(in_file)      # load laz file
            all_xyz = (np.vstack((laz_file.x, laz_file.y, laz_file.z)).T.astype(np.float32))        # get xyz
            all_x, all_y, all_z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
            # all_r, all_g, all_b = np.asarray(laz_file.red), np.asarray(laz_file.green), np.asarray(laz_file.blue)     # get color if available
            # all_r, all_g, all_b = np.ceil((all_r / 65536) * 255), np.ceil((all_g / 65536) * 255), np.ceil((all_b / 65536) * 255)
            all_intensity = np.asarray(laz_file.intensity)      # get intensity if available
            all_labels = laz_file.label     # get labels if available
        elif args.data_format == 'ply':
            data = read_ply(str(in_file))   # load ply file
            all_xyz = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32) # get xyz
            all_x, all_y, all_z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
            # all_r, all_g, all_b = data['red'], data['green'], data['blue']        # get color if available
            all_intensity = data['intensity']        # get intensity if available
            all_labels = data['class']          # get labels if available
        elif args.data_format == 'txt':
            # load txt file, need to change the code according to your data format
            data = np.loadtxt(str(in_file))
            all_xyz = data[:, :3].astype(np.float32)        # get xyz
            all_x, all_y, all_z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
            all_r, all_g, all_b = data[:, 3], data[:, 4], data[:, 5]         # get color if available
            all_intensity = data[:, 6]      # get intensity if available
            all_labels = data[:, -1]        # get labels if available
        else:
            raise Exception('data format not supported')

        # remap labels
        if remap:
            config_path = 'CONFIG_FILE'        # change to your own config file
            all_labels = remap_label(config_path, all_labels)

        # compute point normals if not in original tile
        if add_normals:
            object_pcd = o3d.geometry.PointCloud()
            points = np.stack((all_x, all_y, all_z), axis=-1)
            object_pcd.points = o3d.utility.Vector3dVector(points)
            object_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.matrix.round(np.array(object_pcd.normals), 2)
            normals_z = normals[:, 2]

            normals_of_interest = np.squeeze(np.where(normals_z <= -0.95))
            mean_z = np.mean(all_z[normals_of_interest])
            normal_indxs_to_change = (
                        (all_z[normals_of_interest] > (mean_z - 2)) & (all_z[normals_of_interest] < (mean_z + 2)))
            normals_z[normals_of_interest[normal_indxs_to_change]] = np.absolute(
                normals_z[normals_of_interest[normal_indxs_to_change]])
            normals[:, 2] = normals_z
            all_normals_x, all_normals_y, all_normals_z = normals[:, 0], normals[:, 1], normals[:, 2]

        # crop tile around object of interest and apply augmentation methods
        pnt_idxs = np.squeeze(np.where(all_labels == class_label))
        if pnt_idxs.shape[0] == 0:
            continue
        coordinates = all_xyz[pnt_idxs]
        clustering = DBSCAN().fit(coordinates)
        print('tilecode: {}'.format(tilecode))
        print('unique clusters with label {}: {}'.format(class_label, len(np.unique(clustering.labels_))))
        if len(np.unique(clustering.labels_)) > 0:

            for item_id, cluster_label in enumerate(np.unique(clustering.labels_)):
                if random.uniform(0, 1) < reproduce_percentage:
                    next_tile = False

                    # Precalculate cropped tile dimensions
                    cluster_point_idxs = np.where(clustering.labels_ == cluster_label)
                    cluster_points = coordinates[cluster_point_idxs]
                    mean_x, mean_y = round(np.mean(cluster_points[:, 0]), 2), round(np.mean(cluster_points[:, 1]), 2)

                    min_x_cropped_tile = max(mean_x - tile_radius, min(all_x))
                    max_x_cropped_tile = min(mean_x + tile_radius, max(all_x))
                    min_y_cropped_tile = max(mean_y - tile_radius, min(all_y))
                    max_y_cropped_tile = min(mean_y + tile_radius, max(all_y))
                    condition = ((all_x >= min_x_cropped_tile) & (all_x <= max_x_cropped_tile) & (
                                all_y >= min_y_cropped_tile) & (all_y <= max_y_cropped_tile))
                    condition_indxs = np.asarray(condition).nonzero()[0]

                    # Crop tile
                    xyz_cropped = all_xyz[condition_indxs]
                    # r_cropped, g_cropped, b_cropped = all_r[condition_indxs], all_g[condition_indxs], all_b[
                    #     condition_indxs]
                    intensity_cropped = all_intensity[condition_indxs]
                    labels_cropped = all_labels[condition_indxs]
                    if add_normals:
                        normals_x_cropped = all_normals_x[condition_indxs]
                        normals_y_cropped = all_normals_y[condition_indxs]
                        normals_z_cropped = all_normals_z[condition_indxs]
                    _, cluster_point_idxs_cropped, _ = np.intersect1d(list(condition_indxs),
                                                                      pnt_idxs[np.squeeze(cluster_point_idxs)],
                                                                      return_indices=True)

                    while next_tile == False:
                        xyz = xyz_cropped
                        # r, g, b = r_cropped, g_cropped, b_cropped
                        intensity = intensity_cropped
                        labels = labels_cropped

                        # recovery original labels
                        # TODO: add recovery of original labels, and check if this is necessary

                        if add_normals:
                            normals_x, normals_y, normals_z = normals_x_cropped, normals_y_cropped, normals_z_cropped

                        # translate complete tile with predefined translation factor
                        if translation:
                            xyx, x_value, y_value = translate_point_cloud(xyz, translation_factor_limits)
                            value = str(x_value) + '_' + str(y_value)

                        # scale complete tile with predefined scaling factor
                        if scaling:
                            xyz, value = scale_point_cloud(xyz, scaling_factor_limits)

                        # rotate complete tile with predefined degree limit
                        if cloud_rotation:
                            xyz, value = rotate_point_cloud(xyz, degree_limit_tile_rotation)

                        # Generate and save tile
                        if manual_check:
                            object_pcd = o3d.geometry.PointCloud()
                            points = np.stack((xyz[:, 0], xyz[:, 2], xyz[:, 1]), axis=-1)
                            object_pcd.points = o3d.utility.Vector3dVector(points)
                            # colors = (np.dstack((r, g, b))[0] /255)
                            # object_pcd.colors = o3d.utility.Vector3dVector(colors)
                            o3d.visualization.draw_geometries([object_pcd])
                            save = input('save [y/n]: ')
                        else:
                            save = 'y'

                        if save == 'y':
                            next_tile = True
                            save_file_path = args.file_path / 'argument_data' /augm_type
                            save_file_path.mkdir(parents=True, exist_ok=True)
                            if args.data_format == 'las':
                                cropped_laz_file = laspy.create(file_version="1.2", point_format=3)
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                    name="label", type="uint8", description="Labels"))
                                cropped_laz_file.x, cropped_laz_file.y, cropped_laz_file.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                                cropped_laz_file.red, cropped_laz_file.green, cropped_laz_file.blue = r, g, b
                                cropped_laz_file.intensity = intensity
                                cropped_laz_file.label = labels

                                # add normals
                                if add_normals:
                                    cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                        name="normal_x", type="float", description="normal_x"))
                                    cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                        name="normal_y", type="float", description="normal_y"))
                                    cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                        name="normal_z", type="float", description="normal_z"))
                                    cropped_laz_file.normal_x = normals_x
                                    cropped_laz_file.normal_y = normals_y
                                    cropped_laz_file.normal_z = normals_z

                                # save tile with information in file name
                                cropped_laz_file.write(str(save_file_path) + "/{}_{}_{}_{}_{}.laz".format(
                                    tilecode, augm_type, class_name, item_id, tile_radius))
                                # cropped_laz_file.write(str(save_file_path) + "/{}_{}_{}.laz".format(tilecode, augm_type, value))
                            elif args.data_format == 'ply':
                                # save croped file into ply format
                                # cropped_ply_file = str(save_file_path) + '/{}_{}_{}_{}_{}.ply'.format(
                                #     tilecode, augm_type, class_name, item_id, tile_radius)
                                cropped_ply_file = str(save_file_path) + '/{}_{}_All_{}.ply'.format(tilecode, augm_type, item_id)
                                if add_normals:
                                    # write_ply(cropped_ply_file, [xyz, r, g, b, intensity, labels, normals_x, normals_y, normals_z],
                                    #           ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class', 'nx', 'ny', 'nz'])
                                    write_ply(cropped_ply_file,[xyz, intensity, labels, normals_x, normals_y, normals_z],
                                              ['x', 'y', 'z', 'intensity', 'class', 'nx', 'ny', 'nz'])
                                else:
                                    # write_ply(cropped_ply_file, [xyz, r, g, b, intensity, labels],
                                    #           ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class'])
                                    write_ply(cropped_ply_file, [xyz, intensity, labels],['x', 'y', 'z', 'intensity', 'class'])

                        elif save == 'n':
                            next_tile = True

def tradition_argument_tile(tilecodes, args, argument_id = 10, translation=False, scaling=False, cloud_rotation=False,
                             add_normals=True, manual_check=True, remap=False):

    tile_radius = args.tile_radius  # radius of tile.
    translation_factor_limits = args.translation_factor_limits
    scaling_factor_limits = args.scaling_factor_limits  # world scaling factor to scale complete tile.
    degree_limit_tile_rotation = args.degree_limit_tile_rotation # maximum rotation degree of complete tile.
    reproduce_percentage = args.reproduce_percentage  # what probability to make new tile out of object of interest. range 0-1.
    augm_type = args.augm_type
    total = 0

    for i, in_file in enumerate(tilecodes):

        tilecode = in_file.stem

        # ************* load default point cloud ***************
        if args.data_format == 'las':
            laz_file = laspy.read(in_file)      # load laz file
            all_xyz = (np.vstack((laz_file.x, laz_file.y, laz_file.z)).T.astype(np.float32))        # get xyz
            all_x, all_y, all_z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
            all_r, all_g, all_b = np.asarray(laz_file.red), np.asarray(laz_file.green), np.asarray(laz_file.blue)     # get color if available
            all_r, all_g, all_b = np.ceil((all_r / 65536) * 255), np.ceil((all_g / 65536) * 255), np.ceil((all_b / 65536) * 255)
            all_intensity = np.asarray(laz_file.intensity)      # get intensity if available
            all_labels = laz_file.label     # get labels if available
        elif args.data_format == 'ply':
            data = read_ply(str(in_file))   # load ply file
            all_xyz = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32) # get xyz
            all_x, all_y, all_z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
            all_r, all_g, all_b = data['red'], data['green'], data['blue']        # get color if available
            all_intensity = data['intensity']        # get intensity if available
            all_labels = data['class']          # get labels if available
        elif args.data_format == 'txt':
            # load txt file, need to change the code according to your data format
            data = np.loadtxt(str(in_file))
            all_xyz = data[:, :3].astype(np.float32)        # get xyz
            all_x, all_y, all_z = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
            all_r, all_g, all_b = data[:, 3], data[:, 4], data[:, 5]         # get color if available
            all_intensity = data[:, 6]      # get intensity if available
            all_labels = data[:, -1]        # get labels if available
        else:
            raise Exception('data format not supported')

        # remap labels
        if remap:
            config_path = 'semantic-Vienna.yaml'        # change to your own config file
            all_labels = remap_label(config_path, all_labels)

        # compute point normals if not in original tile
        if add_normals:
            object_pcd = o3d.geometry.PointCloud()
            points = np.stack((all_x, all_y, all_z), axis=-1)
            object_pcd.points = o3d.utility.Vector3dVector(points)
            object_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            normals = np.matrix.round(np.array(object_pcd.normals), 2)
            normals_z = normals[:, 2]

            normals_of_interest = np.squeeze(np.where(normals_z <= -0.95))
            mean_z = np.mean(all_z[normals_of_interest])
            normal_indxs_to_change = (
                        (all_z[normals_of_interest] > (mean_z - 2)) & (all_z[normals_of_interest] < (mean_z + 2)))
            normals_z[normals_of_interest[normal_indxs_to_change]] = np.absolute(
                normals_z[normals_of_interest[normal_indxs_to_change]])
            normals[:, 2] = normals_z
            all_normals_x, all_normals_y, all_normals_z = normals[:, 0], normals[:, 1], normals[:, 2]

        print('tilecode: {}'.format(tilecode))
        for item_id in range(argument_id):
            if random.uniform(0, 1) < reproduce_percentage:
                next_tile = False

                while next_tile == False:

                    # translate complete tile with predefined translation factor
                    if translation:
                        xyx, x_value, y_value = translate_point_cloud(all_xyz, translation_factor_limits)
                        value = str(x_value) + '_' + str(y_value)

                    # scale complete tile with predefined scaling factor
                    if scaling:
                        xyz, value = scale_point_cloud(all_xyz, scaling_factor_limits)

                    # rotate complete tile with predefined degree limit
                    if cloud_rotation:
                        xyz, value = rotate_point_cloud(all_xyz, degree_limit_tile_rotation)

                    # Generate and save tile
                    if manual_check:
                        object_pcd = o3d.geometry.PointCloud()
                        points = np.stack((xyz[:, 0], xyz[:, 2], xyz[:, 1]), axis=-1)
                        object_pcd.points = o3d.utility.Vector3dVector(points)
                        colors = np.stack((all_r, all_g, all_b), axis=-1)
                        object_pcd.colors = o3d.utility.Vector3dVector(colors / 256)
                        o3d.visualization.draw_geometries([object_pcd])
                        save = input('save [y/n]: ')
                    else:
                        save = 'y'

                    if save == 'y':
                        next_tile = True
                        save_file_path = args.file_path / 'argument_data' /augm_type
                        save_file_path.mkdir(parents=True, exist_ok=True)
                        if args.data_format == 'las':
                            cropped_laz_file = laspy.create(file_version="1.2", point_format=3)
                            cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                name="label", type="uint8", description="Labels"))
                            cropped_laz_file.x, cropped_laz_file.y, cropped_laz_file.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                            cropped_laz_file.red, cropped_laz_file.green, cropped_laz_file.blue = all_r / 255 * 65536, all_g / 255 * 65536, all_b / 255 * 65536
                            cropped_laz_file.intensity = all_intensity
                            cropped_laz_file.label = all_labels

                            # add normals
                            if add_normals:
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                    name="normal_x", type="float", description="normal_x"))
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                    name="normal_y", type="float", description="normal_y"))
                                cropped_laz_file.add_extra_dim(laspy.ExtraBytesParams(
                                    name="normal_z", type="float", description="normal_z"))
                                cropped_laz_file.normal_x = all_normals_x
                                cropped_laz_file.normal_y = all_normals_y
                                cropped_laz_file.normal_z = all_normals_z

                            # save tile with information in file name
                            cropped_laz_file.write(str(save_file_path) + "/{}_{}_{}.laz".format(tilecode, augm_type, item_id))
                        elif args.data_format == 'ply':
                            # save croped file into ply format
                            cropped_ply_file = str(save_file_path) + '/{}_{}_All_{}.ply'.format(tilecode, augm_type, item_id)
                            if add_normals:
                                write_ply(cropped_ply_file, [xyz, all_r, all_g, all_b, all_intensity, all_labels, all_normals_x, all_normals_y, all_normals_z],
                                          ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class', 'nx', 'ny', 'nz'])
                            else:
                                write_ply(cropped_ply_file, [xyz, all_r, all_g, all_b, all_intensity, all_labels],
                                          ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class'])

                    elif save == 'n':
                        next_tile = True


def translate_point_cloud(coordinates, translation_factor_limits):
    translation_factor_x = round(random.uniform(translation_factor_limits[0], translation_factor_limits[1]), 1)
    translation_factor_y = round(random.uniform(translation_factor_limits[0], translation_factor_limits[1]), 1)
    print('translation x: {}, translation y: {}'.format(translation_factor_x, translation_factor_y))
    return coordinates + [translation_factor_x, translation_factor_y, 0], translation_factor_x, translation_factor_y


def scale_point_cloud(coordinates, scaling_factor_limits):
    mean_x, mean_y = round(np.mean(coordinates[:, 0]), 2), round(np.mean(coordinates[:, 1]), 2)
    scaling_factor = 1
    while scaling_factor == 1:
        scaling_factor = np.round(random.uniform(scaling_factor_limits[0], scaling_factor_limits[1]), 2)
    print('scaling factor: {}'.format(scaling_factor))
    translated_coordinates = coordinates - [mean_x, mean_y, 0]
    scaled_coordinates = translated_coordinates * [scaling_factor, scaling_factor, scaling_factor]
    return scaled_coordinates + [mean_x, mean_y, 0], scaling_factor


def rotate_point_cloud(coordinates, degree_limit_tile_rotation):
    degrees = random.randint(0, degree_limit_tile_rotation)
    print('point cloud rotation degrees: {}'.format(degrees))
    phi = degrees * np.pi / 180
    rotation_matrix = [[math.cos(phi), -math.sin(phi), 0],
                       [math.sin(phi), math.cos(phi), 0],
                       [0, 0, 1]]
    mean_x, mean_y = round(np.mean(coordinates[:, 0]), 2), round(np.mean(coordinates[:, 1]), 2)
    translated_coordinates = coordinates - [mean_x, mean_y, 0]
    rotated_coordinates = (translated_coordinates @ rotation_matrix) + [mean_x, mean_y, 0]
    return rotated_coordinates, degrees


def rotate_object(coordinates, cluster_point_idxs, degree_limit_object_rotation):
    degrees = random.randint(-degree_limit_object_rotation, degree_limit_object_rotation)
    print('object rotation degrees: {}'.format(degrees))
    phi = degrees * np.pi / 180
    rotation_matrix = [[math.cos(phi), -math.sin(phi), 0],
                       [math.sin(phi), math.cos(phi), 0],
                       [0, 0, 1]]
    cluster_points = coordinates[cluster_point_idxs]
    mean_x, mean_y = round(np.mean(cluster_points[:, 0]), 2), round(np.mean(cluster_points[:, 1]), 2)
    translated_cluster_points = cluster_points - [mean_x, mean_y, 0]
    transformed_cluster_points = (translated_cluster_points @ rotation_matrix) + [mean_x, mean_y, 0]
    coordinates[np.squeeze(cluster_point_idxs)] = transformed_cluster_points
    return coordinates, degrees

def remap_label(config_file, label):
    # function to remap lable into a new classes
    '''
    :param label: N*1 label
    :param config_file: the config file path
    :return: remapped label: N*1 label
    '''
    with open(config_file, 'r') as f:
        DATA = yaml.safe_load(f)
    remap_dict = DATA['learning_map']

    for i, value in enumerate(remap_dict):
        label[label == value] = remap_dict[value]
    return label

# set HongKong datasets augment
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=Path, default='DATATSET_PATH',
                        help='The dataset file path')
    parser.add_argument('--dataset', type=str, default='HongKong', help='The datasets name')
    parser.add_argument('--data_format', type=str, default='ply', help='The datasets format')
    parser.add_argument('--tile_radius', type=int, default=10**6, help='radius of tile for MOBA, set to 10**6 if MOBA shoule not be uesd')
    parser.add_argument('--translation_factor_limits', default=[-100,100], help='maximum translation in x and y direction')
    parser.add_argument('--scaling_factor_limits', default=[0.75, 1.25], help='world scaling factor to scale complete tile')
    parser.add_argument('--degree_limit_tile_rotation', default=360, help='maximum rotation degree of complete tile')
    parser.add_argument('--degree_limit_object_rotation', default=180, help='maximum rotation degree of objetc of interest')
    parser.add_argument('--augm_type', type=str, default='ALL', help='augmentation type. should be set as ROTATION, TRANSLATION, SCALING or MOBA, depending on configuration.')
    parser.add_argument('--reproduce_percentage', default=1, help='probability to make new tile out of object of interest. range 0-1.')

    parser.add_argument('--Translation', action='store_true', help='whether to use translation')
    parser.add_argument('--Rotation', action='store_true', help='whether to use rotation')
    parser.add_argument('--Scaling', action='store_true', help='whether to use scaling')
    parser.add_argument('--add_normals', action='store_true', help='whether to add normals')
    parser.add_argument('--manual_check', action='store_true', help='whether to use manual check')
    args = parser.parse_args()

    class_names = 'CLASS_NAMES'
    class_label = 6  # label of object of interest to crop tile for.

    args.class_names = class_names
    args.class_label = class_label

    tilecodes = list(args.file_path.glob('*.{}'.format(args.data_format)))

    # generate_augmented_tiles(tilecodes, args, translation=True, scaling=True,
    #                          cloud_rotation=True, add_normals=True, manual_check=False, remap=True)

    tradition_argument_tile(tilecodes, args, translation=True, scaling=True,cloud_rotation=True, add_normals=True, manual_check=False, remap=False)


