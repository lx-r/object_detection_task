#!/usr/bin/env python

""" Show 3D point cloud

    Author: Huanle Zhang
    Website: www.huanlezhang.com
"""

import numpy as np
import open3d as o3d

def show_point_cloud(pc_data):
    # pc_data: point cloud data
    vis = o3d.visualization
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc_data[:, :-1])
    pc_color = np.zeros([len(pc_data), 3])
    max_i = np.max(pc_data[:, -1])
    min_i = np.min(pc_data[:, -1])
    pc_color[:, -1] = (pc_data[:, -1] - min_i) / max_i
    cloud.colors = o3d.utility.Vector3dVector(pc_color)
    downsample_cloud = cloud.random_down_sample(0.3)
    obb_box3d = cloud.get_oriented_bounding_box()
    center = cloud.get_center()
    max_bound = cloud.get_max_bound()
    min_bound = cloud.get_min_bound()
    print(center, max_bound, min_bound)
    vis.draw([downsample_cloud, obb_box3d])

if __name__ == '__main__':
    path_to_point_cloud = "/media/lx/data/code/mmdetection3d/demo/data/kitti/velo/000008.bin"
    point_cloud_data = np.fromfile(path_to_point_cloud, '<f4')  # little-endian float32
    point_cloud_data = np.reshape(point_cloud_data, (-1, 4))  # x, y, z, r
    show_point_cloud(point_cloud_data)
