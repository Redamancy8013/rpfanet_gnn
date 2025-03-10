#!/usr/bin/env python3

from logging import raiseExceptions
import os
import json
import rospy
import time
import numpy as np
import math
from numpyencoder import NumpyEncoder
from pyquaternion import Quaternion
from utils.utils import nms, create_experiment_folder, initiate_submission_file, gen_measurement_of_this_class, initiate_classification_submission_file, readout_parameters
from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target as pmbmgnn_tracker
from trackers.PMBMGNN import util as pmbmgnn_ulti
from datetime import datetime
# from evaluate.util.utils import TrackingConfig, config_factory
# from evaluate.evaluate_tracking_result import TrackingEval
# import multiprocessing
# import argparse
from tqdm import tqdm
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs.msg as sensor_msgs
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Int32, Bool, Float32
import std_msgs.msg as std_msgs
from geometry_msgs.msg import PoseStamped, Pose, Point
from detr.msg import DetectedObject, DetectedObjectList


result_file = '/home/ez/project/detr/result'
now = datetime.now()
formatedtime = now.strftime("%Y-%m-%d-%H-%M-%S")
out_file_directory_for_this_experiment = create_experiment_folder(result_file, formatedtime, 'tracking')



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

    _cos = math.cos(theta)
    _sin = math.sin(theta)

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


def object_to_marker_labels(box, frame_id="base", marker_id=None, duration=0.15, color=None):

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
    # marker.text = box['tracking_id'] + ' ' + str(box['tracking_score'])
    marker.text = box['tracking_id']


    marker.action = marker.ADD
    marker.frame_locked = True
    marker.lifetime = rospy.Duration.from_sec(duration)
    marker.scale.x, marker.scale.y,marker.scale.z = 0.8, 0.8, 0.8

    marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 1.0
    # marker.color.a = float(box['tracking_score'])
    marker.color.a = 1.0

    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = center[0], center[1], center[2] + 2

    marker.pose.orientation.w = 1.0

    marker.lifetime = rospy.Duration.from_sec(duration)
    return marker


def object_to_marker(box, frame_id="base", marker_id=None, duration=0.15, color=None):

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

    marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
    marker.color.a = 1.0 if np.isnan(box['tracking_score']) else float(box['tracking_score'])
    marker.points = line_points_from_3d_bbox(
                        center[0], center[1], center[2],
                        box['size'][1], box['size'][2], box['size'][0], Quaternion(orientation).yaw_pitch_roll[0])

    marker.lifetime = rospy.Duration.from_sec(duration)
    return marker


class GnnPmbNode:
    def __init__(self):
        rospy.init_node("GnnPmbNode")
        rospy.loginfo("Starting GnnPmbNode.")

        config='/home/ez/project/detr/src/detr/scripts/configs/gnnpmb_parameters.json'

        self.image = Image()
        self.pointclouds = PointCloud2()   
        self.flag = 0   # 判断前一帧是否有检测信息
        self.framecount = 0
        self.update_frequency = 8
        self.classifications = ['car']
        self.pre_timestamp = time.time()

        self.count = 0

        with open(config, 'r') as f:
            self.parameters=json.load(f)

        self.gnnpmb_filter = {}
        self.birth_rate = {}
        self.detection_score_thr = {}
        self.nms_score = {}
        self.confidence_score = {}
        self.filter_pruned = {}
        self.filter_model = {}
        self.filter_predicted = {}
        self.filter_updated = {}

        for classification in self.classifications:
            birth_rate, P_s, P_d, use_ds_as_pd,clutter_rate, bernoulli_gating, extraction_thr, ber_thr, poi_thr, eB_thr, detection_score_thr, nms_score, confidence_score, P_init = readout_parameters(classification, self.parameters)
            self.filter_model[classification] = pmbmgnn_ulti.gen_filter_model(clutter_rate,P_s,P_d, classification, extraction_thr, ber_thr, poi_thr, eB_thr,bernoulli_gating, use_ds_as_pd, P_init)        
            self.birth_rate[classification] = birth_rate
            self.detection_score_thr[classification] = detection_score_thr
            self.nms_score[classification] = nms_score
            self.confidence_score[classification] = confidence_score
            self.filter_pruned[classification] = []
            self.filter_predicted[classification] = {}
            self.filter_updated[classification] = {}

        # 初始化发布节点
        self.publishers = {}
        self.publishers['bboxes'] = rospy.Publisher('bboxes_track', MarkerArray, queue_size=10, latch=True)       # 目标边界框
        self.publishers['labels'] = rospy.Publisher('labels_track', MarkerArray, queue_size=10, latch=True)       # 目标信息：id、置信度、速度等
     
        rospy.Subscriber("/detections", DetectedObjectList, self.bboxes_rec, queue_size=100)


    def bboxes_rec(self, boxes):

        start = time.time()
        # while not self.flag_callback3:
        #     rospy.sleep(0.1)
        
        # self.flag_callback3 = False
        
        Z_k_all = {}
        for classification in self.classifications:
            Z_k_all[classification] = []

        timestamp = (boxes.header.stamp.secs) + (boxes.header.stamp.nsecs / 1e9)

        frame_id = boxes.header.frame_id
        
        for i, box in enumerate(boxes.objects):

            bbox = {}

            if i == 0:
                sample_token = box.sample_token

            bbox['sample_token'] = box.sample_token
            bbox['translation'] = list(box.translation)
            bbox['size'] = list(box.size)
            bbox['rotation'] = list(box.rotation)
            bbox['velocity'] = list(box.velocity)
            bbox['detection_name'] = box.detection_name
            bbox['detection_score'] = box.detection_score


            for classification in self.classifications:
                if bbox['detection_name'] == classification:
                    if bbox['detection_score'] > self.detection_score_thr[classification]:
                        Z_k_all[classification].append(bbox)
                        self.count += 1


        print(f'Number of objects = {self.count}')

        if self.count  == 0:
            self.flag = 0
            self.ids = 0

        else:

            if self.flag == 0:
                self.pre_timestamp = (boxes.header.stamp.secs) + (boxes.header.stamp.nsecs / 1e9)
            
            cur_timestamp = timestamp
            time_lag = cur_timestamp - self.pre_timestamp
            giou_gating = -0.5


            for classification in self.classifications:

              
                classification_submission = {}
                classification_submission['results']={}
                classification_submission['results'][sample_token] = []

                
                result_indexes = nms(Z_k_all[classification], threshold=self.nms_score[classification])
                Z_k=[]
                for idx in result_indexes:
                    Z_k.append(Z_k_all[classification][idx])

                if self.flag == 0:
                    self.gnnpmb_filter[classification] = pmbmgnn_tracker.PMBMGNN_Filter(self.filter_model[classification])
                    self.filter_predicted[classification] = self.gnnpmb_filter[classification].predict_initial_step(Z_k, self.birth_rate[classification])

                else:

                    if self.framecount == 20:
                        self.framecount = 0
                        self.gnnpmb_filter[classification] = pmbmgnn_tracker.PMBMGNN_Filter(self.filter_model[classification])
                        self.filter_predicted[classification] = self.gnnpmb_filter[classification].predict_initial_step(Z_k, self.birth_rate[classification])

                    else:
                        self.framecount += 1
                        self.filter_predicted[classification] = self.gnnpmb_filter[classification].predict(time_lag,self.filter_pruned[classification], Z_k, self.birth_rate[classification])

                
                self.filter_updated[classification] = self.gnnpmb_filter[classification].update(Z_k, self.filter_predicted[classification], self.confidence_score[classification],giou_gating)



                if classification == 'pedestrian':
                    if len(Z_k)==0:
                        estimatedStates_for_this_classification = self.gnnpmb_filter[classification].extractStates_with_custom_thr(self.filter_updated[classification], 0.7)
                    else:
                        estimatedStates_for_this_classification = self.gnnpmb_filter[classification].extractStates(self.filter_updated[classification])
                else:
                    estimatedStates_for_this_classification = self.gnnpmb_filter[classification].extractStates(self.filter_updated[classification])


                # 发布3D边界框及其ID
                self.markerArray = MarkerArray()
                self.markers = MarkerArray()
                new =[]
                
                for idx in range(len(estimatedStates_for_this_classification['mean'])):
                    instance_info = {}                    
                   

                    instance_info['sample_token'] = sample_token
                    translation_of_this_target = [estimatedStates_for_this_classification['mean'][idx][0][0],
                                                estimatedStates_for_this_classification['mean'][idx][1][0], estimatedStates_for_this_classification['elevation'][idx]]
                    
                    
                    # global frame
                    center = np.array(translation_of_this_target)
                    orientation = np.array(estimatedStates_for_this_classification['rotation'][idx])

                    instance_info['translation'] = center.tolist()
                    instance_info['size'] = estimatedStates_for_this_classification['size'][idx]
                    instance_info['rotation'] = list(orientation)
                    instance_info['velocity'] = [estimatedStates_for_this_classification['mean']
                                                [idx][2][0], estimatedStates_for_this_classification['mean'][idx][3][0]]
                    instance_info['tracking_id'] = estimatedStates_for_this_classification['classification'][idx]+'_'+str(
                        estimatedStates_for_this_classification['id'][idx])
                    
                    instance_info['tracking_name'] = estimatedStates_for_this_classification['classification'][idx]
                    instance_info['tracking_score']=estimatedStates_for_this_classification['detection_score'][idx]

                    temp1 = object_to_marker(instance_info, frame_id=frame_id, marker_id=idx, duration= 3.2 / self.update_frequency)
                    temp2 = object_to_marker_labels(instance_info, frame_id=frame_id, marker_id=idx, duration= 3.2 / self.update_frequency)

                    self.markers.markers.append(temp1)
                    self.markerArray.markers.append(temp2)
                    new.append(instance_info)
                    
                estimatedStates_for_this_classification = new
                classification_submission['results'][sample_token] = estimatedStates_for_this_classification

                self.publishers['bboxes'].publish(self.markers)
                self.publishers['labels'].publish(self.markerArray)

                self.filter_pruned[classification] = self.gnnpmb_filter[classification].prune(self.filter_updated[classification])


                self.pre_timestamp = cur_timestamp

                self.count = 0
                
                # with open(out_file_directory_for_this_experiment+'/{}_{}.json'.format(sample_token, classification), 'w') as f:
                #     json.dump(classification_submission, f, cls=NumpyEncoder)

            self.flag = 1

            end = time.time()

            print(f'Tracking_Duration = {end - start}')


if __name__ == '__main__':
    ros_node = GnnPmbNode()
    rospy.spin()

    
    
   