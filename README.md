# rpfanet_gnn

## Notice
Before you compile this project, you need to put the rosbag into the proper directory.

Put the rosbag named 2024-04-16-16-27-12.bag into /home/ez/project/detr/bag/2024-04-16-16-27-12.bag

## Environment

conda env create -f environment.yml

conda activate OpenPCDet

## Compile

## Run

cd /home/ez/project/detr/

source devel/setup.bash

### Detect

roslaunch detr detect.launch

### Track

roslaunch detr tracking.launch
