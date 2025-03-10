#!/usr/bin/env python3
# license removed for brevity
import os
import rospy 
import numpy as np
import cv2
import time
from nuscenes.nuscenes import NuScenes
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs.msg as sensor_msgs

from visualization_msgs.msg import Marker, MarkerArray

from std_msgs.msg import String, Int32, Bool, Float32
import std_msgs.msg as std_msgs

from geometry_msgs.msg import PoseStamped, Pose, Point

from tracking_module.msg import DetectedObject, DetectedObjectList

import math
from math import sin, cos
from logging import raiseExceptions
import json
from numpyencoder import NumpyEncoder
from datetime import datetime
from pyquaternion import Quaternion
import argparse


def compute_pose(translation, rotation):
    """Compute Pose from translation and rotation

    Args:
        translation (List[float]): x, y, z
        rotation (List[float]): w, x, y, z

    Returns:
        geometry_msgs.msg.Pose
    """    
    msg = Pose()
    msg.orientation.w = rotation[0]
    msg.orientation.x = rotation[1]
    msg.orientation.y = rotation[2]
    msg.orientation.z = rotation[3]

    msg.position.x    = translation[0]
    msg.position.y    = translation[1]
    msg.position.z    = translation[2]
    return msg


def array2pc2(points, parent_frame, field_names='xyza'):
    """ Creates a point cloud message.
    Args:
        points: Nxk array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
        field_names : name for the k channels repectively i.e. "xyz" / "xyza"
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate(field_names)]

    header = std_msgs.Header(frame_id=parent_frame, stamp= rospy.Time.now())

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * len(field_names)),
        row_step=(itemsize * len(field_names) * points.shape[0]),
        data=data
    )


def line_points_from_3d_bbox(x, y, z, w, h, l, theta):

    corner_matrix = np.array(
        [[-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [ 1,  1,  1],
        [ 1, -1,  1],
        [-1, -1,  1],
        [-1,  1,  1],
        [-1,  1, -1]], dtype=np.float32
    )
    relative_eight_corners = 0.5 * corner_matrix * np.array([l, w, h]) #[8, 3]

    _cos = cos(theta)
    _sin = sin(theta)

    rotated_corners_x, rotated_corners_y = (
            relative_eight_corners[:, 0] * _cos +
                -relative_eight_corners[:, 1] * _sin,
        relative_eight_corners[:, 0] * _sin +
            relative_eight_corners[:, 1] * _cos
        ) #[8]
    rotated_corners = np.stack([rotated_corners_x, rotated_corners_y, relative_eight_corners[:,2]], axis=-1) #[8, 3]
    abs_corners = rotated_corners + np.array([x, y, z])  # [8, 3]

    points = []
    for i in range(1, 5):
        points += [
            Point(x=abs_corners[i, 0], y=abs_corners[i, 1], z=abs_corners[i, 2]),
            Point(x=abs_corners[i%4+1, 0], y=abs_corners[i%4+1, 1], z=abs_corners[i%4+1, 2])
        ]
        points += [
            Point(x=abs_corners[(i + 4)%8, 0], y=abs_corners[(i + 4)%8, 1], z=abs_corners[(i + 4)%8, 2]),
            Point(x=abs_corners[((i)%4 + 5)%8, 0], y=abs_corners[((i)%4 + 5)%8, 1], z=abs_corners[((i)%4 + 5)%8, 2])
        ]
    points += [
        Point(x=abs_corners[2, 0], y=abs_corners[2, 1], z=abs_corners[2, 2]),
        Point(x=abs_corners[7, 0], y=abs_corners[7, 1], z=abs_corners[7, 2]),
        Point(x=abs_corners[3, 0], y=abs_corners[3, 1], z=abs_corners[3, 2]),
        Point(x=abs_corners[6, 0], y=abs_corners[6, 1], z=abs_corners[6, 2]),

        Point(x=abs_corners[4, 0], y=abs_corners[4, 1], z=abs_corners[4, 2]),
        Point(x=abs_corners[5, 0], y=abs_corners[5, 1], z=abs_corners[5, 2]),
        Point(x=abs_corners[0, 0], y=abs_corners[0, 1], z=abs_corners[0, 2]),
        Point(x=abs_corners[1, 0], y=abs_corners[1, 1], z=abs_corners[1, 2])
    ]

    return points


def object_to_marker_labels(box, lidar_data, nusc, frame_id="base", marker_id=None, duration=0.15, color=None):

    nuscenes_dir = '/home/ez/project/dataset/nuscenes'
    lidar_path = os.path.join(nuscenes_dir, lidar_data['filename'])

    calib_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    ego_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # global frame
    center = np.array(box['translation'])
    orientation = np.array(box['rotation'])

    marker = Marker()

    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id
    marker.type = marker.TEXT_VIEW_FACING
    if marker_id is not None:
        marker.id = marker_id

    vel =  math.sqrt(box['velocity'][0]**2 + box['velocity'][1]**2)
    # marker.text = box['tracking_id'] + ',score:' + str(box['tracking_score']) + ',vel:' + str(vel)
    # marker.text = box['detection_name'] + ',score:' + str(box['detection_score'])
    marker.text = box['detection_name'] + ' ' + str(box['detection_score'])


    marker.action = marker.ADD
    marker.frame_locked = True
    marker.lifetime = rospy.Duration.from_sec(duration)
    marker.scale.x, marker.scale.y,marker.scale.z = 0.8, 0.8, 0.8

    marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
    marker.color.a = float(box['detection_score'])

    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center[0], center[1], center[2] + 2

    marker.pose.orientation.w = 1.0

    marker.lifetime = rospy.Duration.from_sec(duration)
    return marker


def object_to_marker(box, lidar_data, nusc, frame_id="base", marker_id=None, duration=0.15, color=None):

    nuscenes_dir = '/home/ez/project/dataset/nuscenes'
    lidar_path = os.path.join(nuscenes_dir, lidar_data['filename'])


    # global frame
    center = np.array(box['translation'])
    orientation = np.array(box['rotation'])


    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = frame_id

    if marker_id is not None:
        marker.id = marker_id
    marker.type = 5 
    marker.scale.x = 0.05
    marker.pose.orientation.w = 1.0

    detection_name = box['detection_name']
    marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 1.0
    marker.color.a = 1.0 if np.isnan(box['detection_score']) else float(box['detection_score'])
    marker.points = line_points_from_3d_bbox(
                        center[0], center[1], center[2],
                        box['size'][1], box['size'][2], box['size'][0], Quaternion(orientation).yaw_pitch_roll[0])

    marker.lifetime = rospy.Duration.from_sec(duration)
    return marker


def publish_image(image, image_publisher, camera_info_publisher, P, frame_id):
    """Publish image and info message to ROS.

    Args:
        image: numpy.ndArray.
        image_publisher: rospy.Publisher
        camera_info_publisher: rospy.Publisher, should publish CameraInfo
        P: projection matrix [3, 4]. though only [3, 3] is useful.
        frame_id: string, parent frame name.
    """
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
    image_msg.header.frame_id = frame_id
    image_msg.header.stamp = rospy.Time.now()
    image_publisher.publish(image_msg)

    camera_info_msg = CameraInfo()
    camera_info_msg.header.frame_id = frame_id
    camera_info_msg.header.stamp = rospy.Time.now()
    camera_info_msg.height = image.shape[0]
    camera_info_msg.width = image.shape[1]
    camera_info_msg.D = [0, 0, 0, 0, 0]
    camera_info_msg.K = np.reshape(P[0:3, 0:3], (-1)).tolist()
    P_no_translation = np.zeros([3, 4])
    P_no_translation[0:3, 0:3] = P[0:3, 0:3]
    camera_info_msg.P = np.reshape(P_no_translation, (-1)).tolist()

    camera_info_publisher.publish(camera_info_msg)


def publish_point_cloud(pointcloud, pc_publisher, frame_id, field_names='xyza'):
    """Convert point cloud array to PointCloud2 message and publish
    
    Args:
        pointcloud: point cloud array [N,3]/[N,4]
        pc_publisher: ROS publisher for PointCloud2
        frame_id: parent_frame name.
        field_names: name for each channel, ['xyz', 'xyza'...]
    """
    msg = array2pc2(pointcloud, frame_id, field_names)
    pc_publisher.publish(msg)


class PubInfoNode:
    def __init__(self):
        rospy.init_node("PubInfoNode")
        rospy.loginfo("Starting PubInfoNode.")


        # 读取参数
        self.read_params()
        self.scene_index = 0
        self.count = 0


        # 打开并读取目标检测结果文件
        with open(self.detection_file, 'rb') as f:
            inference_meta = json.load(f)
        self.inference = inference_meta['results']
 
        
        # 初始化发布节点
        self.publishers = {}
        self.publishers['bboxes'] = rospy.Publisher('bboxes', MarkerArray, queue_size=10, latch=True)       # 目标边界框
        self.publishers['labels'] = rospy.Publisher('labels', MarkerArray, queue_size=10, latch=True)       # 目标信息：id、置信度、速度等
        self.publishers['camera'] = rospy.Publisher('image', Image, queue_size=10, latch=True)               # 图像数据
        self.publishers['lidar'] = rospy.Publisher('pointclouds', PointCloud2, queue_size=10, latch=True)    # 点云数据
        self.publishers['camera_pose'] = rospy.Publisher('camera_pose', PoseStamped, queue_size=10, latch=True)
        self.publishers['camera_info'] = rospy.Publisher('camera_info', CameraInfo, queue_size=10, latch=True)
        self.publishers['lidar_pose'] = rospy.Publisher('lidar_pose', PoseStamped, queue_size=10, latch=True)
        self.publishers['ego_pose'] = rospy.Publisher('ego_pose', PoseStamped, queue_size=10, latch=True)
        self.publishers['currentest'] = rospy.Publisher('estimate', DetectedObjectList, queue_size=10, latch=True)
 

        # 读取nuscenes数据集
        self.nusc = NuScenes(version=self.nuscenes_version, dataroot=self.nuscenes_dir, verbose=True)
        self.current_scene = self.nusc.scene[self.scene_index]
        self.current_sample = self.nusc.get('sample', self.current_scene['first_sample_token']) 
        print(f"Switch to scenes {self.scene_index},frame: {self.current_sample['token']} ")

        # 设置发布频率
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.update_frequency), self.publish_callback)
        

        
    def read_params(self):

        # 图像、雷达元数据
        self.nuscenes_dir = rospy.get_param("~NUSCENES_DIR", '/home/ez/project/dataset')
        self.nuscenes_version = rospy.get_param("~NUSCENES_VER", 'v1.0-mini')

        # 更新频率
        self.update_frequency = float(rospy.get_param("~UPDATE_FREQUENCY", 8.0))

        # 目标检测结果
        self.detection_file = rospy.get_param("~DETECTION_FILE", '/home/ez/project/detection/pointpillars-val.json')


    def _camera_publish(self, camera_data, is_publish_image=False):
          
        cs_record   = self.nusc.get("calibrated_sensor", camera_data['calibrated_sensor_token'])
        ego_record  = self.nusc.get("ego_pose", camera_data['ego_pose_token'])

        image_path = os.path.join(self.nuscenes_dir, camera_data['filename'])
        imsize    = (camera_data["width"], camera_data["height"])
        channel = camera_data['channel']
        
        # 发布摄像头位姿
        cam_intrinsic = np.array(cs_record["camera_intrinsic"]) #[3 * 3]
        rotation = cs_record["rotation"] #list, [4] r, x, y, z
        translation = cs_record['translation']
        relative_pose = compute_pose(translation, rotation)
        
        msg = PoseStamped()
        msg.pose = relative_pose
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time.now()
        self.publishers['camera_pose'].publish(msg)

        if is_publish_image:

            image = cv2.imread(image_path)
            publish_image(image,
                        self.publishers['camera'],
                        self.publishers['camera_info'],
                        cam_intrinsic,
                        channel)

    def _lidar_publish(self, lidar_data, is_publish_lidar=False):
        
        cs_record   = self.nusc.get("calibrated_sensor", lidar_data['calibrated_sensor_token'])
        ego_record  = self.nusc.get("ego_pose", lidar_data['ego_pose_token'])

        lidar_path = os.path.join(self.nuscenes_dir, lidar_data['filename'])
        channel = 'LIDAR_TOP'
        
        # 发布雷达位姿
        cam_intrinsic = np.array(cs_record["camera_intrinsic"]) #[3 * 3]
        rotation = cs_record["rotation"] #list, [4] r, x, y, z
        translation = cs_record['translation']
        relative_pose = compute_pose(translation, rotation)
        
        msg = PoseStamped()
        msg.pose = relative_pose
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time.now()
        self.publishers['lidar_pose'].publish(msg)

        # 发布载体位姿
        ego_rotation = ego_record["rotation"]
        ego_translation = ego_record["translation"]
        frame_location = compute_pose(ego_translation, ego_rotation)
        
        msg = PoseStamped()
        msg.pose = frame_location
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        self.publishers['ego_pose'].publish(msg)

        # 发布点云数据
        if is_publish_lidar:

            point_cloud = np.fromfile(os.path.join(self.nuscenes_dir, lidar_data['filename']), dtype=np.float32).reshape(-1, 5)[:, :4]
            publish_point_cloud(point_cloud, self.publishers['lidar'], "LIDAR_TOP")
              
    def publish_callback(self, event):

        # 发布摄像头图像及信息
        # channels = ['CAM_BACK',  'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
        channels = ['CAM_FRONT_RIGHT']

        # 发布图像
        for channel in channels:
            data = self.nusc.get('sample_data', self.current_sample['data'][channel])
            self._camera_publish(data, is_publish_image=True)

        # 发布点云
        data = self.nusc.get('sample_data', self.current_sample['data']['LIDAR_TOP'])
        self._lidar_publish(data, is_publish_lidar=True)


        data = self.nusc.get('sample_data', self.current_sample['data']['LIDAR_TOP'])

        

        # 发布3D边界框及其ID
        self.markerArray = MarkerArray()
        self.markers = MarkerArray()
        self.currentest = DetectedObjectList()


        calib_data = self.nusc.get('calibrated_sensor', data['calibrated_sensor_token'])
        ego_data = self.nusc.get('ego_pose', data['ego_pose_token'])


        if self.current_sample['token'] in self.inference.keys():
            estimated_bboxes_at_current_frame = self.inference[self.current_sample['token']]

            for i, box in enumerate(estimated_bboxes_at_current_frame):

                bbox = DetectedObject()

                bbox.sample_token = box['sample_token']
                bbox.translation = box['translation']
                bbox.size = box['size']
                bbox.rotation = box['rotation']
                bbox.velocity = box['velocity']
                bbox.detection_name = box['detection_name']
                bbox.detection_score = box['detection_score']
                bbox.attribute_name = box['attribute_name']

                if not self.count:

                    # global frame
                    center = np.array(box['translation'])
                    orientation = np.array(box['rotation'])

                    # 从global frame转换到ego vehicle frame
                    quaternion = Quaternion(ego_data['rotation']).inverse
                    center -= np.array(ego_data['translation'])
                    center = np.dot(quaternion.rotation_matrix, center)
                    orientation = quaternion * orientation

                    # 从ego vehicle frame转换到lidar frame
                    quaternion = Quaternion(calib_data['rotation']).inverse
                    center -= np.array(calib_data['translation'])
                    center = np.dot(quaternion.rotation_matrix, center)
                    orientation = quaternion * orientation

                    box['translation'] = center.tolist()
                    box['rotation'] = list(orientation)


                temp1 = object_to_marker(box, data, self.nusc, frame_id='LIDAR_TOP', marker_id=i, duration= 1.6 / self.update_frequency)
                temp2 = object_to_marker_labels(box, data, self.nusc, frame_id='LIDAR_TOP', marker_id=i, duration= 1.6 / self.update_frequency)

                self.markers.markers.append(temp1)
                self.markerArray.markers.append(temp2)
                self.currentest.objects.append(bbox)
            

        self.publishers['bboxes'].publish(self.markers)
        # print(len(self.markers.markers))
        self.publishers['labels'].publish(self.markerArray)

        self.currentest.header.stamp = rospy.Time.now()
        self.currentest.header.frame_id = 'LIDAR_TOP'
        self.publishers['currentest'].publish(self.currentest)


        

        if (self.current_sample['next'] == ''):
            # If end reached, loop back from the start
            self.scene_index = self.scene_index + 1
            if self.scene_index == 10:
                self.scene_index = 0
                self.count = self.count + 1
            self.current_scene = self.nusc.scene[self.scene_index]
            self.current_sample = self.nusc.get('sample', self.current_scene['first_sample_token']) 
            print(f"Switch to scenes {self.scene_index},frame: {self.current_sample['token']} ")
        else:
            self.current_sample = self.nusc.get('sample', self.current_sample['next'])
            print(f"Switch to scenes {self.scene_index},frame: {self.current_sample['token']} ")


if __name__ == "__main__":
    ros_node = PubInfoNode()
    rospy.spin()