import tkinter as tk
import subprocess

root = tk.Tk()

root.geometry("500x250")  # 调整宽度和高度
root.title("4D Millimeter-Wave Radar Point Cloud Detection and Tracking Software")

# Add labels for the software name
title_label_line1 = tk.Label(root, text="4D Millimeter-Wave Radar Point Cloud", font=("Helvetica", 14))
title_label_line1.grid(row=0, column=0, columnspan=2, pady=(10, 0))

title_label_line2 = tk.Label(root, text="Object Detection and Tracking Software (v1.0)", font=("Helvetica", 14))
title_label_line2.grid(row=1, column=0, columnspan=2, pady=(0, 10))

def visualize_single_frame():
    subprocess.Popen(['gnome-terminal', '--', 'python3', '/path/to/single_frame_visualization.py'])

def visualize_realtime():
    subprocess.Popen(['gnome-terminal', '--', 'python3', '/path/to/realtime_visualization.py'])

def object_detection():
    subprocess.Popen(['gnome-terminal', '--', 'python3', '/path/to/object_detection.py'])

def object_tracking():
    subprocess.Popen(['gnome-terminal', '--', 'python3', '/path/to/object_tracking.py'])

button_width = 20
button_height = 2

single_frame_button = tk.Button(root, text="Single Frame Visualization", command=visualize_single_frame, width=button_width, height=button_height)
single_frame_button.grid(row=2, column=0, padx=10, pady=10)

realtime_button = tk.Button(root, text="Real-time Visualization", command=visualize_realtime, width=button_width, height=button_height)
realtime_button.grid(row=2, column=1, padx=10, pady=10)

detection_button = tk.Button(root, text="Object Detection", command=object_detection, width=button_width, height=button_height)
detection_button.grid(row=3, column=0, padx=10, pady=10)

tracking_button = tk.Button(root, text="Object Tracking", command=object_tracking, width=button_width, height=button_height)
tracking_button.grid(row=3, column=1, padx=10, pady=10)

# 使内容居中
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()