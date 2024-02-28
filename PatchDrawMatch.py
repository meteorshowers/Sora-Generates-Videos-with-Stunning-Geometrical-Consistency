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
import numpy as np
from utils_function import *
from core_function import *



dataset_list = ['sora', "gen2", "pika"]


def PatchDrawMatch(dataset_list):
    for dataset in dataset_list:
        # folder_path = "data/sora"
        folder_path = "data/" + dataset
        images_dirtory = os.path.join(folder_path, "frame_images")
        result_dirtory = os.path.join(folder_path, "image_result")
        # Create output directory if it doesn't exist
        if not os.path.exists(result_dirtory):
            os.makedirs(result_dirtory )


        # Replace with the path to your folder containing MP4 files
        image_dir_lsit = ReadImageFolder(images_dirtory)
        print(image_dir_lsit)
        left_id = 0
        right_id = 30
    
        for image_dir in image_dir_lsit:

            data_name = os.path.basename(image_dir)
            res_dir = os.path.join(result_dirtory , data_name )
            left_img_path  = os.path.join(image_dir, "frame_" + str(left_id) + ".png")
            right_img_path = os.path.join(image_dir, "frame_" + str(right_id) + ".png")
            print(left_img_path)
            print(right_img_path)
            img1 = cv2.imread(left_img_path)
            img2 = cv2.imread(right_img_path)
            img1 = cv2.resize(img1, (1280,720))
            img2 = cv2.resize(img2, (1280,720))

            # 提取正确匹配点的位置
            src_pts = np.float32([])
            dst_pts = np.float32([])
            # Apply perspective transformation to the source points
            matchesMask = np.array([])
            raw_img_matches = DrawMatchingPoints(img1, img2, src_pts, dst_pts, matchesMask )
            raw_image_path = os.path.join(result_dirtory, "raw_" + data_name + "_" + str(left_id) + "_"+ str(right_id) + ".png")
            cv2.imwrite(raw_image_path, raw_img_matches)

            mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts, img_matches= EvaluateErrBetweenTwoImage2(left_img_path, right_img_path, 3)
            image_path = os.path.join(result_dirtory, data_name + "_" + str(left_id) + "_"+ str(right_id) + ".png")
            print(image_path)
            cv2.imwrite(image_path,img_matches)
