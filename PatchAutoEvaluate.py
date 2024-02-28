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



video_list = ["sora", "pika", "gen2"]

def PatchAutoEvaluate(video_list):
    interval_num_list = [60, 50, 40, 30, 20, 10]
    for dataset in video_list:
        for interval_num in interval_num_list:
            # interval_num = 60
            begin_frame = 0 
            end_frame = 71
            ransac_th = 3

            # folder_path = "data/sora"
            folder_path = "data/" + dataset
            images_dirtory = os.path.join(folder_path, "frame_images")
            result_dirtory = os.path.join(folder_path, "full_result")
            brief_result_dirtory = os.path.join(folder_path, dataset + "_brief_result")
            # Create output directory if it doesn't exist
            if not os.path.exists(result_dirtory):
                os.makedirs(result_dirtory )
            if not os.path.exists(brief_result_dirtory):
                os.makedirs(brief_result_dirtory)


            # Replace with the path to your folder containing MP4 files
            image_dir_lsit = ReadImageFolder(images_dirtory)
            print(image_dir_lsit)
            whole_err_list = []
            for image_dir in image_dir_lsit:
                err_list = []
                data_name = os.path.basename(image_dir)
                res_dir = os.path.join(result_dirtory , data_name )
                # Create output directory if it doesn't exist
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir )
                match_inter = int((70 - interval_num) / 10)
                print(match_inter)
                for i in range(0, end_frame, match_inter):
                    if(i + interval_num > end_frame - 1): 
                        break
                    print(i)
                    left_id = i
                    right_id = i + interval_num
                    left_img_path  = os.path.join(image_dir, "frame_" + str(left_id) + ".png")
                    right_img_path = os.path.join(image_dir, "frame_" + str(right_id) + ".png")
                    print(left_img_path)
                    print(right_img_path)

                    mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts= EvaluateErrBetweenTwoImage(left_img_path, right_img_path, ransac_th)
                    err_list.append([i, mean_error, median_error, rmse, mae, keep_rate, num_inliers_F, num_pts])
                print(err_list)
                # 指定CSV文件路径

                info_csv_path = os.path.join(res_dir , str(interval_num) + "_" + str(ransac_th) + "_err.csv")
                # 写入CSV文件
                with open(info_csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["index", "mean_error", "median_error", "rmse", "mae", "keep_rate", "num_inliers_F", "num_pts"])
                    writer.writerows(err_list)
                print(f"CSV文件已创建：{info_csv_path}")

                # cal the brife value

                errors_all = np.array(err_list)
                mean_error = np.mean(errors_all[:, 1])
                median_error = np.median(errors_all[:, 2])
                rmse = np.sqrt(np.mean((errors_all[:, 3] ** 2)))
                mae = np.mean(errors_all[:, 4])
                mean_keep_rate = np.mean(errors_all[:, 5])
                mean_num_inliers_F = np.mean(errors_all[:, 6])
                mean_num_pts = np.mean(errors_all[:, 7])
                whole_err_list.append([data_name, mean_error, median_error, rmse, mae, mean_keep_rate,  mean_num_inliers_F, mean_num_pts])


            whole_err_csv_path = os.path.join(brief_result_dirtory, str(interval_num) + "_" + str(ransac_th) + "_err_whole.csv")
            # 写入CSV文件
            with open(whole_err_csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["data_name", "mean_error", "median_error", "rmse", "mae", "keep_rate", "num_inliers_F", "num_pts"])
                writer.writerows(whole_err_list)
            print(f"CSV文件已创建：{whole_err_csv_path}")
