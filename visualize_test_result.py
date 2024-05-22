#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Partial2Complete 
@File    ：visualize_test_result.py
@IDE     ：PyCharm 
@Author  ：Wang Song 0010 0502尖端遮挡
@Date    ：2024/5/16 上午10:24 
"""
import open3d as o3d
import os
import glob


def visualize_point_cloud(group_number, directory_path):
    # 定义点云文件所在的目录路径和文件的模式
    pattern = os.path.join(directory_path, f'{group_number}_*_normalized_*.ply')

    # 使用glob.glob寻找所有匹配的文件
    files = glob.glob(pattern)

    # 检查是否找到了三个文件
    if len(files) != 3:
        print(f"Found {len(files)} files, expected 3. Please check the naming convention and try again.")
        return

    # 加载点云文件
    gt_cloud_path = [f for f in files if 'gt_label' in f][0]
    partials_cloud_path = [f for f in files if 'partials_label' in f][0]
    pred_cloud_path = [f for f in files if 'pred_label' in f][0]

    gt_cloud = o3d.io.read_point_cloud(gt_cloud_path)
    partials_cloud = o3d.io.read_point_cloud(partials_cloud_path)
    pred_cloud = o3d.io.read_point_cloud(pred_cloud_path)
    # 设置点云颜色
    gt_cloud.paint_uniform_color([1, 0, 0])  # 红色
    partials_cloud.paint_uniform_color([0, 1, 0])  # 绿色
    pred_cloud.paint_uniform_color([0, 0, 1])  # 蓝色
    # 可视化点云
    o3d.visualization.draw_geometries([gt_cloud, partials_cloud, pred_cloud])


while True:
    # 请求用户输入组号
    group_number = input("Enter the group number (e.g., '0000' or '0001'): ")

    # 指定点云文件所在的目录路径
    # 注意这里的'your_directory_path'应该替换成您的实际目录路径
    directory_path = './experiments/P2C/EPN3D_models/test_base_rope/predictions'

    # 调用函数进行可视化
    visualize_point_cloud(group_number, directory_path)
