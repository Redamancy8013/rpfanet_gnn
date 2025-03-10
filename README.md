# Rpfanet_Gnn

This project visualize the real-time object detection and tracking with point clouds only. It uses feature-self-attention mechanism to improve the accuracy of the detection. 

## 1.Notice
Before you compile this project, you need to put the rosbag into the proper directory.

Put the rosbag named `2024-04-16-16-27-12.bag` into `/home/ez/project/detr/bag/`

eg. `/home/ez/project/detr/bag/2024-04-16-16-27-12.bag`

## 2.Environment

`conda env create -f environment.yml`

`conda activate OpenPCDet`

## 3.Compile

## 4.Run

`cd /home/ez/project/detr/`

`source devel/setup.bash`

#### Detect

`roslaunch detr detect.launch`

<div align="center">
  <img src="https://github.com/Redamancy8013/rpfanet_gnn/blob/main/detect.jpg">
</div>

#### Track

`roslaunch detr tracking.launch`

<div align="center">
  <img src="https://github.com/Redamancy8013/rpfanet_gnn/blob/main/track.jpg">
</div>

#### Visualization

This node only visualizes the images and the relevent point clouds real time.

`roslaunch detr visualization.launch`

<div align="center">
  <img src="https://github.com/Redamancy8013/rpfanet_gnn/blob/main/visualization.jpg">
</div>

## 5.More

This project was accomplished on June 1, 2024 and was first upload to the github on March 10, 2025.

Contact Email: 2110539202@qq.com
