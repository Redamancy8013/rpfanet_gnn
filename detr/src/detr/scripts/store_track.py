#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

bridge = CvBridge()
image_counter = 1
bev_counter = 1
previous_counter = 0

def read_image_from_msg(msg):
    return bridge.imgmsg_to_cv2(msg, "bgr8")

def read_point_cloud_from_msg(msg):
    points = []
    for point in pc2.read_points(msg, skip_nans=True):
        points.append([point[0], point[1], point[2]])
    return np.array(points)

def read_bounding_boxes_from_msg(msg):
    bboxes = [list(map(float, line.strip().split())) for line in msg.data.split('\n') if line.strip()]
    return bboxes

def project_to_top_down_view(points, bboxes):
    top_down_view = np.zeros((500, 500, 3), dtype=np.uint8)
    for point in points:
        x, y = int(point[0] * 10), int(point[1] * 10 + 250)
        if 0 <= x < 500 and 0 <= y < 500:
            top_down_view[y, x] = (255, 255, 255)
    
    for bbox in bboxes:
        
        # Draw the rotated bounding box

        for i in range(0, len(bbox), 2):
            start_point = (int(bbox[i] * 10), int(bbox[i + 1] * 10 + 250))
            end_point = (int(bbox[(i + 2) % len(bbox)] * 10), int(bbox[(i + 3) % len(bbox)] * 10 + 250))
            cv2.line(top_down_view, start_point, end_point, (0, 255, 0), 2)
            
    return top_down_view

def save_image(image, save_path):
    # 直接保存图像
    cv2.imwrite(save_path, image)

def process_data(points, bboxes, labels, save_dir):
    global bev_counter
    global image_counter

    top_down_view = project_to_top_down_view(points, bboxes)
    # Mirror the image along the x-axis
    mirrored_image = cv2.flip(top_down_view, 1)
    
    # Rotate the image 90 degrees counterclockwise
    rotated_image = cv2.rotate(mirrored_image, cv2.ROTATE_90_CLOCKWISE)
    
    top_down_view = rotated_image
    
    for bbox, label in zip(bboxes, labels):
        x, y = int((bbox[0] + bbox[2] + bbox[4] + bbox[6]) / 4 * 10 - 20), int((bbox[1] + bbox[3] + bbox[5] + bbox[7]) / 4 * 10 + 230)
        # Mirror the x coordinate
        x = 500 - x
        # Rotate 90 degrees clockwise
        x, y = 500 - y - 10, x - 20
        cv2.putText(top_down_view, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"top_down_view_{bev_counter}.png")
    save_image(top_down_view, save_path)

    bev_counter += 1
    image_counter += 1

def image_callback(msg):
    global image_counter
    global previous_counter
    # 将ROS图像消息转换为OpenCV图像
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    if previous_counter < image_counter:
        save_dir = "/home/ez/project/detr/result/gui/img"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"current_image_{image_counter}.png")
        save_image(cv_image, save_path)
        previous_counter += 1

def point_cloud_callback(msg):
    points = read_point_cloud_from_msg(msg)
    rospy.set_param('/current_points', points.tolist())

def bounding_boxes_callback(msg):
    bboxes = []
    for marker in msg.markers:
        x1, y1, z1 = marker.points[0].x, marker.points[0].y, marker.points[0].z
        x2, y2, z2 = marker.points[1].x, marker.points[1].y, marker.points[1].z
        x3, y3, z3 = marker.points[3].x, marker.points[3].y, marker.points[3].z
        x4, y4, z4 = marker.points[2].x, marker.points[2].y, marker.points[2].z

        bboxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    rospy.set_param('/current_bboxes', bboxes)

def labels_callback(msg):
    labels = [marker.text for marker in msg.markers]
    rospy.set_param('/current_labels', labels)

if __name__ == "__main__":
    rospy.init_node('data_listener', anonymous=True)
    rospy.Subscriber("/raw_img", Image, image_callback)
    rospy.Subscriber("/radar_pc", PointCloud2, point_cloud_callback)
    rospy.Subscriber("/bboxes_track", MarkerArray, bounding_boxes_callback)
    rospy.Subscriber("/labels_track", MarkerArray, labels_callback)
    
    save_dir = "/home/ez/project/detr/result/gui/track/combination"

    rate = rospy.Rate(5)  # 5 Hz
    while not rospy.is_shutdown():

        if rospy.has_param('/current_points') and rospy.has_param('/current_bboxes') and rospy.has_param('/current_labels'):
            
            points = np.array(rospy.get_param('/current_points'))
            bboxes = rospy.get_param('/current_bboxes')
            labels = rospy.get_param('/current_labels')
            
            process_data(points, bboxes, labels, save_dir)
            
            rospy.delete_param('/current_points')
            rospy.delete_param('/current_bboxes')
            rospy.delete_param('/current_labels')
        
        rate.sleep()