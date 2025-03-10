#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from detr.msg import DetectedObject, DetectedObjectList


import time
import numpy as np
import math
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import scipy.linalg as linalg

import numpy as np
from scipy.spatial.transform import Rotation as R

# import sys
# sys.path.append("/home/ez/project/detr/src/pointpillars_ros")

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from detr.msg import DetectedObject, DetectedObjectList



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path: 根目录
            dataset_cfg: 数据集配置
            class_names: 类别名称
            training: 训练模式
            logger: 日志
            ext: 扩展名
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        


def get_indices_and_avg_velocity_in_bbox(points, bbox):
    # Convert the points to the coordinate system of the bbox
    points_in_bbox_frame = points[:, :3] - bbox[:3]
    points_in_bbox_frame[:, 2] -= bbox[5] / 2
    rotation = R.from_euler('z', bbox[6], degrees=False)
    points_in_bbox_frame = rotation.apply(points_in_bbox_frame, inverse=True)

    # Check which points are inside the bbox
    half_dims = bbox[3:6] / 2
    mask = np.all((points_in_bbox_frame >= -half_dims) & (points_in_bbox_frame <= half_dims), axis=1)

    # Get the indices of the points inside the bbox
    indices = np.where(mask)[0]

    # Calculate the average velocity of the points inside the bbox
    avg_velocity = np.mean(points[indices, 3])

    return avg_velocity


class Pointpillars_ROS:
    def __init__(self):
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)
        self.sample_token = 0


    def init_ros(self):
        """ Initialize ros parameters """
        config_path = rospy.get_param("/config_path", "/home/ez/project/detr/src/detr/tools/cfgs/astyx_models/pointpillar_fsa.yaml")
        ckpt_path = rospy.get_param("/ckpt_path", "/home/ez/project/detr/src/detr/tools/output/astyx_models/pointpillar_fsa/astyx_no_kde_fsa59/ckpt/checkpoint_epoch_68.pth")
        # onfig_path = rospy.get_param("/config_path", "/home/ez/project/rp_ros/src/pointpillars_ros/tools/cfgs/astyx_models/pointpillar.yaml")
        # ckpt_path = rospy.get_param("/ckpt_path", "/home/ez/project/rp_ros/src/pointpillars_ros/tools/models/checkpoint_epoch_80.pth")
        self.sub_velo = rospy.Subscriber("/radar_pc", PointCloud2, self.lidar_callback, queue_size=1,  buff_size=2**12)
        self.pub_bbox = rospy.Publisher("/detections", DetectedObjectList, queue_size=10)
        return config_path, ckpt_path


    def init_pointpillars(self, config_path, ckpt_path):
        """ Initialize second model """
        logger = common_utils.create_logger() # 创建日志
        logger.info('-----------------Quick Demo of Pointpillars-------------------------')
        cfg_from_yaml_file(config_path, cfg)  # 加载配置文件
        
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            ext='.bin', logger=logger
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        # 加载权重文件
        self.model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
        self.model.cuda() # 将网络放到GPU上
        self.model.eval() # 开启评估模式


    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix


    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """
        # pcl_msg = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        pcl_msg = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "velocity", "intensity"))
        np_p = np.array(list(pcl_msg), dtype=np.float32)
        # # 旋转轴
        # rand_axis = [0,1,0]
        # #旋转角度0.1047
        # yaw = 0
        # #返回旋转矩阵
        # rot_matrix = self.rotate_mat(rand_axis, yaw)
        # np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T
        
        # convert to xyzi point cloud
        # x = np_p_rot[:, 0].reshape(-1)
        # y = np_p_rot[:, 1].reshape(-1)
        # z = np_p_rot[:, 2].reshape(-1)
        x = np_p[:, 0].reshape(-1)
        y = np_p[:, 1].reshape(-1)
        z = np_p[:, 2].reshape(-1)
        if np_p.shape[1] >= 4: # if intensity field exists
            i = np_p[:, 4].reshape(-1)
        else:
            i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        points = np.stack((x, y, z, i)).T
        # 组装数组字典
        input_dict = {
            'points': points,
            'frame_id': msg.header.frame_id,
        }
        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict) # 数据预处理
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict) # 将数据放到GPU上
        pred_dicts, _ = self.model.forward(data_dict) # 模型前向传播
        scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()
        mask = scores > 0.0
        scores = scores[mask]
        boxes_lidar = pred_dicts[0]['pred_boxes'][mask].detach().cpu().numpy()
        label = pred_dicts[0]['pred_labels'][mask].detach().cpu().numpy()
        num_detections = boxes_lidar.shape[0]

        rospy.loginfo("The num is: %d ", num_detections)

        # print(boxes_lidar)
        # print(scores)
        # print(label)

        arr_bbox = DetectedObjectList()

        for i in range(num_detections):

            avg_velocity = get_indices_and_avg_velocity_in_bbox(np_p, boxes_lidar[i])

            bbox = DetectedObject()
            bbox.sample_token = str(self.sample_token)
            bbox.translation = np.array([boxes_lidar[i][0], boxes_lidar[i][1], boxes_lidar[i][2] + boxes_lidar[i][5] / 2], dtype=np.float32)
            bbox.size = np.array([boxes_lidar[i][3], boxes_lidar[i][4], boxes_lidar[i][5]], dtype=np.float32)
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.rotation = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            bbox.velocity = np.array([avg_velocity * math.cos(boxes_lidar[i][6]), avg_velocity * math.sin(boxes_lidar[i][6])], dtype=np.float32)
            if label[i] == 1:
                bbox.detection_name = 'car'
            else:
                bbox.detection_name = 'others'
            bbox.detection_score = scores[i]

            arr_bbox.objects.append(bbox)
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        
        self.pub_bbox.publish(arr_bbox)
        self.sample_token += 1


if __name__ == '__main__':
    sec = Pointpillars_ROS()
    rospy.init_node('pointpillars_ros_node', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
