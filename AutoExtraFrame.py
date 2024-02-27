#############################################################################
# code for paper: Sora Generateds Video with Stunning Geometical Consistency
# arxiv: https://arxiv.org/abs/
# Author: <NAME> xuanyili
# email: xuanyili.edu@gmail.com
# github: https://github.com/meteorshowers/SoraGeoEvaluate
#############################################################################
import os
import cv2
import csv
from utils_function import *

#input

video_list = ["sora", "pika", "gen2"]


def AutoExtraFrame(video_list):
    for video in video_list:
            
        folder_path = "data/{}".format(video)
        images_dirtory = os.path.join(folder_path, "frame_images")
        # Create output directory if it doesn't exist
        if not os.path.exists(images_dirtory):
            os.makedirs(images_dirtory)
        # Replace with the path to your folder containing MP4 files
        video_lsit = ReadMp4FilesInFolder(folder_path)
        print(video_lsit)
        frame_interval = 1
        video_info = []
        for video_path in video_lsit:
            name = GetFileName(video_path)
            frame_dirtory = os.path.join(images_dirtory, name)
            print(frame_dirtory)

            total_frames, fps, time = ExtractFrames(video_path, frame_dirtory, frame_interval)
            video_info.append([name, total_frames, fps, time])
        # 指定CSV文件路径
        info_csv_path = os.path.join(folder_path, "info.csv")
        # 写入CSV文件
        with open(info_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["name", "total_frames", "fps", "time"])
            writer.writerows(video_info)
        print(f"CSV文件已创建：{info_csv_path}")


