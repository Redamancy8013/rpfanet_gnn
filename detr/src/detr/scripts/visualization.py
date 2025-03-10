import numpy as np
import argparse
import open3d as o3d


def read_point_cloud(file_path):
    # 跳过文件的前两行（列标题）
    data = np.loadtxt(file_path, skiprows=2)
    return data

def visualize_point_cloud_open3d(data):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :3])
    
    # 将点云颜色设置为白色
    colors = np.ones((data.shape[0], 3))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='4D Radar Point Cloud Visualization')
    vis.add_geometry(point_cloud)
    
    # 设置背景颜色为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize 4D Radar Point Cloud')
    parser.add_argument('file_path', type=str, help='Path to the point cloud data file')
    args = parser.parse_args()

    point_cloud_data = read_point_cloud(args.file_path)
    visualize_point_cloud_open3d(point_cloud_data)
