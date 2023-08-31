#!/usr/bin/env python

""" Draw labeled objects in images

    Author: Huanle Zhang
    Website: www.huanlezhang.com
"""

import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import os
import open3d as o3d
import open3d.visualization as o3d_vis
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


MARKER_COLOR = {
    'Car': [1, 0, 0],               # red
    'DontCare': [0, 0, 0],          # black
    'Pedestrian': [0, 0, 1],        # blue
    'Van': [1, 1, 0],               # yellow
    'Cyclist': [1, 0, 1],           # magenta
    'Truck': [0, 1, 1],             # cyan
    'Misc': [0.5, 0, 0],            # maroon
    'Tram': [0, 0.5, 0],            # green
    'Person_sitting': [0, 0, 0.5]}  # navy

# image border width
BOX_BORDER_WIDTH = 5

# point size
POINT_SIZE = 0.005


def show_object_in_image(img_filename, label_filename):
    img = mping.imread(img_filename)
    with open(label_filename) as f_label:
        lines = f_label.readlines()
        for line in lines:
            line = line.strip('\n').split()
            left_pixel, top_pixel, right_pixel, bottom_pixel = [int(float(line[i])) for i in range(4, 8)]
            box_border_color = MARKER_COLOR[line[0]]
            for i in range(BOX_BORDER_WIDTH):
                img[top_pixel+i, left_pixel:right_pixel, :] = box_border_color
                img[bottom_pixel-i, left_pixel:right_pixel, :] = box_border_color
                img[top_pixel:bottom_pixel, left_pixel+i, :] = box_border_color
                img[top_pixel:bottom_pixel, right_pixel-i, :] = box_border_color
    plt.imshow(img)
    plt.show()

def show_object_in_point_cloud(point_cloud_filename, label_filename, calib_filename):
    pc_data = np.fromfile(point_cloud_filename, '<f4')  # little-endian float32
    pc_data = np.reshape(pc_data, (-1, 4))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc_data[:,:-1])
    pc_color = np.ones((len(pc_data), 3))
    calib = load_kitti_calib(calib_filename)
    rot_axis = 2
    with open(label_filename) as f_label:
        lines = f_label.readlines()
        bboxes_3d = []
        for line in lines:
            line = line.strip('\n').split()
            point_color = MARKER_COLOR[line[0]]
            veloc, dims, rz, box3d_corner = camera_coordinate_to_point_cloud(line[8:15], calib['Tr_velo_to_cam'])
            bboxes_3d.append(np.concatenate((veloc, dims, np.array([rz]))))
        bboxes_3d = np.array(bboxes_3d)
        print(bboxes_3d.shape)
        lines = []
        for i in range(len(bboxes_3d)):
            center = bboxes_3d[i, 0:3]
            dim = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bboxes_3d[i, 6]
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
            # bottom center to gravity center
            center[rot_axis] += dim[rot_axis] / 2

            box3d = o3d.geometry.OrientedBoundingBox(center, rot_mat, dim)
            
            line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(
                box3d)
            line_set.paint_uniform_color(np.array(point_color) / 255.)
            lines.append(line_set)
        
        for i, v in enumerate(pc_data):
            if point_in_cube(v[:3], box3d_corner) is True:
                pc_color[i, :] = point_color
                
        cloud.colors = o3d.utility.Vector3dVector(pc_color)
        o3d_vis.draw([*lines, cloud])

def point_in_cube(point, cube):
    z_min = np.amin(cube[:, 2], 0)
    z_max = np.amax(cube[:, 2], 0)

    if point[2] > z_max or point[2] < z_min:
        return False

    point = Point(point[:2])
    polygon = Polygon(cube[:4, :2])

    return polygon.contains(point)


def load_kitti_calib(calib_file):
    """
    This script is copied from https://github.com/AI-liu/Complex-YOLO
    """
    with open(calib_file) as f_calib:
        lines = f_calib.readlines()

    P0 = np.array(lines[0].strip('\n').split()[1:], dtype=np.float32)
    P1 = np.array(lines[1].strip('\n').split()[1:], dtype=np.float32)
    P2 = np.array(lines[2].strip('\n').split()[1:], dtype=np.float32)
    P3 = np.array(lines[3].strip('\n').split()[1:], dtype=np.float32)
    R0_rect = np.array(lines[4].strip('\n').split()[1:], dtype=np.float32)
    Tr_velo_to_cam = np.array(lines[5].strip('\n').split()[1:], dtype=np.float32)
    Tr_imu_to_velo = np.array(lines[6].strip('\n').split()[1:], dtype=np.float32)

    return {'P0': P0, 'P1': P1, 'P2': P2, 'P3': P3, 'R0_rect': R0_rect,
            'Tr_velo_to_cam': Tr_velo_to_cam.reshape(3, 4),
            'Tr_imu_to_velo': Tr_imu_to_velo}


def camera_coordinate_to_point_cloud(box3d, Tr):
    """
    This script is copied from https://github.com/AI-liu/Complex-YOLO
    """
    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype=np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2

        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle
        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)

    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])

    rz = ry_to_rz(ry)

    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])

    velo_box = np.dot(rotMat, Box)

    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T

    box3d_corner = cornerPosInVelo.transpose()
    dims = np.array([l, w, h])
    # t_lidar: the x, y coordinator of the center of the object
    # box3d_corner: the 8 corners
    print(t_lidar.shape)
    return t_lidar.reshape(-1), dims, rz, box3d_corner.astype(np.float32)


if __name__ == '__main__':

    # updates
    ROOT = "../dataset/sample"
    IMG_DIR = f'{ROOT}/image_2'
    LABEL_DIR = f'{ROOT}/label_2'
    POINT_CLOUD_DIR = f'{ROOT}/velo'
    CALIB_DIR = f'{ROOT}/calib'

    # id for viewing
    file_id = 8

    img_filename = os.path.join(IMG_DIR, '{0:06d}.png'.format(file_id))
    label_filename = os.path.join(LABEL_DIR, '{0:06d}.txt'.format(file_id))
    pc_filename = os.path.join(POINT_CLOUD_DIR, '{0:06d}.bin'.format(file_id))
    calib_filename = os.path.join(CALIB_DIR, '{0:06d}.txt'.format(file_id))

    # show object in image
    show_object_in_image(img_filename, label_filename)

    # show object in point cloud
    show_object_in_point_cloud(pc_filename, label_filename, calib_filename)