#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import open3d as o3d
import rospy
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

class ImagePointCloudViewer:
    def __init__(self, root):
        self.root = root
        self.root.title('Image and Point Cloud Viewer')
        
        self.image_folder_label = Label(root, text='Image Folder: ')
        self.image_folder_label.grid(row=0, column=0, sticky='w')
        
        self.point_cloud_folder_label = Label(root, text='Point Cloud Folder: ')
        self.point_cloud_folder_label.grid(row=0, column=1, sticky='w')
        
        self.image_name_label = Label(root, text='')
        self.image_name_label.grid(row=1, column=0)
        
        self.point_cloud_name_label = Label(root, text='')
        self.point_cloud_name_label.grid(row=1, column=1)
        
        self.image_label = Label(root)
        self.image_label.grid(row=2, column=0)
        
        self.point_cloud_label = Label(root)
        self.point_cloud_label.grid(row=2, column=1)
        
        self.prev_button = Button(root, text='Previous', command=self.show_previous)
        self.prev_button.grid(row=3, column=0)
        
        self.next_button = Button(root, text='Next', command=self.show_next)
        self.next_button.grid(row=3, column=1)
        
        self.image_folder = rospy.get_param('~image_folder', '/home/ez/project/detr/result/gui/img')
        self.point_cloud_folder = rospy.get_param('~point_cloud_folder', '/home/ez/project/detr/result/gui/detect/combination')
        self.image_files = []
        self.point_cloud_files = []
        self.current_index = 0
        self.image_size = (400, 300)  # 调整图像大小

        create_placeholder_images(self.image_folder, 'current_image_0.png')
        create_placeholder_images(self.point_cloud_folder, 'top_down_view_0.png')

        self.update_files()
        rospy.Timer(rospy.Duration(1), self.update_files)  # 每秒更新一次文件列表

    def update_files(self, event=None):
        self.image_folder_label.config(text=f'Image Folder: {self.image_folder}')
        self.point_cloud_folder_label.config(text=f'Point Cloud Folder: {self.point_cloud_folder}')
        
        self.image_files = sorted([f for f in os.listdir(self.image_folder) if (f.endswith('.png') or f.endswith('.jpg'))])
        self.point_cloud_files = sorted([f for f in os.listdir(self.point_cloud_folder) if f.endswith('.png')])
        self.show_image_and_point_cloud()

    def show_image_and_point_cloud(self):
        if self.image_files and self.point_cloud_files:
            image_path = os.path.join(self.image_folder, f"current_image_{self.current_index}.png")
            point_cloud_path = os.path.join(self.point_cloud_folder, f"top_down_view_{self.current_index}.png")
            
            self.show_image(image_path)
            self.show_point_cloud(point_cloud_path)
            
            self.image_name_label.config(text=f"current_image_{self.current_index}.png")
            self.point_cloud_name_label.config(text=f"top_down_view_{self.current_index}.png")

    def show_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)  # 调整图像大小
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def show_point_cloud(self, point_cloud_path):
        image = cv2.imread(point_cloud_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)  # 调整图像大小
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.point_cloud_label.config(image=image)
        self.point_cloud_label.image = image

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image_and_point_cloud()

    def show_next(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image_and_point_cloud()

def create_placeholder_images(folder, filename):
    placeholder_image = np.zeros((300, 400, 3), dtype=np.uint8)
    placeholder_image.fill(255)  # Fill with white color
    cv2.putText(placeholder_image, 'No Image', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(folder, filename), placeholder_image)

if __name__ == '__main__':
    rospy.init_node('image_point_cloud_viewer')
    root = Tk()
    viewer = ImagePointCloudViewer(root)
    root.mainloop()
    rospy.spin()
