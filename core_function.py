#############################################################################
# code for paper: Sora Generateds Video with Stunning Geometical Consistency
# arxiv: https://arxiv.org/abs/
# Author: <NAME> xuanyili
# email: xuanyili.edu@gmail.com
# github: https://github.com/meteorshowers/SoraGeoEvaluate
#############################################################################
import cv2
import csv
import os
import numpy as np

def compute_fundamental_error(inliers, src_pts, dst_pts, F):
    """计算基础矩阵误差"""
    errors = []
    for i in range(len(inliers)):
        if inliers[i] == 0:
            continue
        pt1 = np.array([src_pts[i][0][0], src_pts[i][0][1], 1.0])
        pt2 = np.array([dst_pts[i][0][0], dst_pts[i][0][1], 1.0])
        epiline = np.dot(F, pt1)
        error = abs(np.dot(epiline, pt2)) / np.sqrt(epiline[0]**2 + epiline[1]**2)
        errors.append(error)
    errors = np.array(errors)
    # Calculate Mean Error (ME)
    mean_error = np.mean(errors)
    # Calculate Median Error
    median_error = np.median(errors)
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((errors ** 2)))
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(errors)

    # Calculate Standard Deviation of Errors
    std_deviation_errors = np.std(errors)
    # Print the results

    return mean_error, median_error, rmse, mae

def EvaluateErrBetweenTwoImage(left_img_path, right_img_path, ransac_th):
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    #  align the size the all picture
    img1 = cv2.resize(img1, (1280,720))
    img2 = cv2.resize(img2, (1280,720))

    # 创建SIFT特征点检测器
    num_corners = 40000
    sift = cv2.SIFT_create(num_corners)

    # 在两张图片上检测SIFT特征点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(len(kp1))
    print(len(kp2))

    # 使用FLANN算法匹配特征点
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # 通过距离比率去除错误匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    # 提取正确匹配点的位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 匹配点对数
    num_pts = len(src_pts)

    # 使用基础矩阵RANSAC去除误匹配
    F, inliers_F = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, ransac_th)
    num_inliers_F = np.sum(inliers_F)
    # print("FundamentalMat:\n", F / F[2][2])
    # 计算误差
    mean_error, median_error, rmse, mae = compute_fundamental_error(inliers_F, src_pts, dst_pts, F)
    keep_rate = num_inliers_F/num_pts
    print(f"Mean Error (ME): {mean_error}")
    print(f"Median Error: {median_error}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"基础矩阵RANSAC去除误匹配后正确点数：{num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")

    return mean_error, median_error, rmse, mae, keep_rate , num_inliers_F, num_pts






def percent_linear_stretch(img, percent):
    p_low, p_high = np.percentile(img, (percent, 100 - percent))
    print("gray : ", p_low, p_high)
    img_rescale = (img - p_low) / (p_high - p_low) * 255
    return img_rescale


def StereoMathing(img1, img2, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 创建SIFT特征点检测器
    num_corners = 100000
    # sift = cv2.SIFT_create(num_corners)

    # sift = cv2.SIFT_create(nfeatures=100000, nOctaveLayers=1, contrastThreshold=0.001, edgeThreshold=5, sigma=1.6)
    sift = cv2.SIFT_create(nfeatures=40000,nOctaveLayers=3,contrastThreshold=0.005,edgeThreshold=30)

    # 在两张图片上检测SIFT特征点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print("number of img1 key points", len(kp1))
    print("number of img2 key points", len(kp2))

    # 使用FLANN算法匹配特征点
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # 通过距离比率去除错误匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # 提取正确匹配点的位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 使用基础矩阵RANSAC去除误匹配
    F, inliers_F = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, 0.5)


    # 计算极线校正矩阵
    # img_size = (img1.shape[1], img1.shape[0] )
    # img_size = (1440, 2048)
    img_size = (2048, 1440)
    _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, img_size)

    image_width, image_height = img1.shape[0], img1.shape[1]   # 图像宽度和高度
    # corrected_H, projected_width, projected_height = correct_H(H1, image_width, image_height)


    # 极线校正
    img1_rectified = cv2.warpPerspective(img1, H1, img_size)
    img2_rectified = cv2.warpPerspective(img2, H2, img_size)

    left = right = 0
    top = bottom = 0
    color = [0, 0, 0]
    img1_rectified = cv2.copyMakeBorder(img1_rectified, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    img2_rectified = cv2.copyMakeBorder(img2_rectified, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    # 显示并保存校正后的影像
    cv2.imwrite(os.path.join(out_dir,'image1_rectified.png'), img1_rectified)
    cv2.imwrite(os.path.join(out_dir,'image2_rectified.png'), img2_rectified)


    print("开始稠密匹配")
    # 转换为灰度图
    gray_left = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # SGBM 参数
    window_size = 13
    # opencv sgbm只能是16的倍数
    min_disp = -96
    num_disp = 192
    block_size = window_size
    uniqueness_ratio = 10
    speckle_window_size = 400
    speckle_range = 32
    p1 = 4 * window_size ** 2
    p2 =  128 * window_size ** 2
    pre_filter_cap = 63
    # 创建 SGBM 对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        # uniquenessRatio=uniqueness_ratio,
        # speckleWindowSize=speckle_window_size,
        # speckleRange=speckle_range,
        disp12MaxDiff=1,
        P1= p1,
        P2= p2,
        # preFilterCap = pre_filter_cap
    )


    # 计算视差图
    disp_map_32 = np.float32(stereo.compute(gray_left, gray_right))
    disp_map_32[disp_map_32 < (min_disp + 10) * 16] = 0

    # 创建左右两张图的mask
    mask_left = np.zeros_like(gray_left)
    mask_left[gray_left != 0] = 1.0
    mask_right = np.zeros_like(gray_right)
    mask_right[gray_right != 0] = 1.0

    disp_map_32  = disp_map_32 * mask_left
    disp_map_32  = disp_map_32 * mask_right
    disp_map_32[disp_map_32 == 0] = 0

    disp_map_32 = disp_map_32 * 0.0625
    cv2.imwrite(os.path.join(out_dir, "disp_map.tif"),disp_map_32)




    H_inv = np.linalg.inv(H1)
    img_unwarped = cv2.warpPerspective(disp_map_32, H_inv, ((img1.shape[1], img1.shape[0] )))

    img_band = percent_linear_stretch(img_unwarped, 0.05)
    img_8bit = img_band.astype('uint8')

    cv2.imwrite(os.path.join(out_dir, "disp_map_8.png"),img_8bit)

    img_unwarped[img_unwarped == 0] = np.nan
    cv2.imwrite(os.path.join(out_dir, "left.png"), img1)
    cv2.imwrite(os.path.join(out_dir, "right.png"), img2)
    cv2.imwrite(os.path.join(out_dir, "depth.tif"), img_unwarped)



def DrawMatchingPoints(img1, img2, pts1, pts2, inliers):
    print(len(pts1))
    print(len(pts2))
    print(len(inliers))
    # Create a blank image for drawing lines between matching points
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img3[:h1, :w1 , :] = img1
    img3[:h2, w1:w1 + w2, :] = img2


    # Draw lines between matching points
    for pt1, pt2, flag in zip(pts1, pts2, inliers):
        
        print(pt1[0][0],pt1[0][1])
        x1, y1 = map(int, (int(pt1[0][0]), int(pt1[0][1])))
        x2, y2 = map(int, (int(pt2[0][0]), int(pt2[0][1])))
        if flag ==1 :
            cv2.line(img3, (x1, y1), (x2 + w1, y2), (0, 255, 0), 1)
        else:
            cv2.line(img3, (x1, y1), (x2 + w1, y2), (0, 0, 255), 1)

    # # Show the image with matching points
    # cv2.namedWindow('Matching Points', 0)
    # cv2.imshow('Matching Points', img3)
    return img3    




def EvaluateErrBetweenTwoImage2(left_img_path, right_img_path, ransac_th):
    img1 = cv2.imread(left_img_path)
    img2 = cv2.imread(right_img_path)

    #  align the size the all picture
    img1 = cv2.resize(img1, (1280,720))
    img2 = cv2.resize(img2, (1280,720))

    # 创建SIFT特征点检测器
    num_corners = 10000
    # sift = cv2.SIFT_create(num_corners)
    sift = cv2.SIFT_create(nfeatures=num_corners,nOctaveLayers=3,contrastThreshold=0.0003,edgeThreshold=30)

    # 在两张图片上检测SIFT特征点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print(len(kp1))
    print(len(kp2))

    # 使用FLANN算法匹配特征点
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # 通过距离比率去除错误匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 提取正确匹配点的位置
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 匹配点对数
    num_pts = len(src_pts)

    # 使用基础矩阵RANSAC去除误匹配
    F, inliers_F = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, ransac_th)
    num_inliers_F = np.sum(inliers_F)
    # print("FundamentalMat:\n", F / F[2][2])
    # 计算误差
    mean_error, median_error, rmse, mae = compute_fundamental_error(inliers_F, src_pts, dst_pts, F)
    keep_rate = num_inliers_F/num_pts
    print(f"Mean Error (ME): {mean_error}")
    print(f"Median Error: {median_error}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"基础矩阵RANSAC去除误匹配后正确点数：{num_inliers_F}/{num_pts} ({num_inliers_F/num_pts*100:.2f}%)")

    # Apply perspective transformation to the source points
    matchesMask = inliers_F.ravel().tolist()
    img_matches = DrawMatchingPoints(img1, img2, src_pts, dst_pts, matchesMask )

    return mean_error, median_error, rmse, mae, keep_rate , num_inliers_F, num_pts, img_matches

