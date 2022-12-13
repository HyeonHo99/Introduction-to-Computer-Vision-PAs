import os
import time
import cv2
import numpy as np
from utils import *


def compute_homography(srcP, destP):
    N = srcP.shape[0]

    A = []
    for i in range(N):
        x, y = srcP[i][0], srcP[i][1]
        _x, _y = destP[i][0], destP[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * _x, y * _x, _x])
        A.append([0, 0, 0, -x, -y, -1, x * _y, y * _y, _y])

    u, s, vh = np.linalg.svd(np.array(A))
    h = vh[-1, :].reshape(3, 3)
    h = h.reshape(3, 3)
    h /= h[2][2]

    return h


def compute_homography_ransac(srcP, destP, th):
    start_time = time.time()
    N = srcP.shape[0]
    srcP_3d = np.concatenate((srcP, np.ones((N, 1))), axis=1)
    inliers = 0
    H = np.identity(3)

    while True:
        random_indices = np.random.choice(N, 4, replace=False)
        _H = compute_homography(srcP[random_indices], destP[random_indices])

        _destP = np.array([(_H @ row)[:2] / (_H @ row)[2] for row in srcP_3d])
        res = destP - _destP
        # test: l1 norm, l2 norm
        norm = np.linalg.norm(res, ord=2, axis=1)
        _inliers = np.sum(norm < th)
        H = _H if _inliers > inliers else H
        inliers = _inliers if _inliers > inliers else inliers

        end_time = time.time()
        if end_time - start_time >= 2.99:
            break
    print(f"\tHomography RANSAC time:{end_time - start_time}")

    return H

if __name__ == "__main__":
    if not os.path.exists("result"):
        os.makedirs("result")

    """
    Feature detection, description, and matching
    """
    print("2-1")

    dirs = ["CV_Assignment_2_Images/cv_desk.png", "CV_Assignment_2_Images/cv_cover.jpg"]
    imgs = [cv2.imread(dir, cv2.IMREAD_GRAYSCALE) for dir in dirs]
    img_desk = imgs[0]
    img_cover = imgs[1]

    # extract ORB features
    orb = cv2.ORB_create()
    kp_desk = orb.detect(img_desk, None)
    kp_desk, des_desk = orb.compute(img_desk, kp_desk)

    kp_cover = orb.detect(img_cover, None)
    kp_cover, des_cover = orb.compute(img_cover, kp_cover)

    match_list = get_match_list(des_desk, des_cover)
    dmatch_list = get_dmatch_list(match_list)
    matched_img = cv2.drawMatches(img_desk, kp_desk, img_cover, kp_cover, dmatch_list, None, flags=2)
    cv2.imshow("2-1", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Computing homography with normalization
    """
    print("2-2")
    match_list = get_match_list_with_threshold(des_desk, des_cover, threshold=0.8)
    N = 15
    srcP, destP = kp2p(src_kp=kp_cover, dest_kp=kp_desk, match_list=match_list, N=N)

    ####### normalize before 'compute_homography' ######
    Ts = normalize_matrix(srcP)
    Td = normalize_matrix(destP)
    srcP_3d = np.concatenate((srcP, np.ones((N, 1))), axis=1)
    destP_3d = np.concatenate((destP, np.ones((N, 1))), axis=1)
    normalized_srcP = np.array([(Ts @ row)[:2] for row in srcP_3d])
    normalized_destP = np.array([(Td @ row)[:2] for row in destP_3d])
    #####################################################

    h = compute_homography(normalized_srcP, normalized_destP)
    # normalize => denormalize
    H = np.linalg.inv(Td) @ h @ Ts

    transformed = cv2.warpPerspective(img_cover, H, (img_desk.shape[1], img_desk.shape[0]))
    cv2.imshow('homography with normalization (1)', transformed)
    combined = combine_with_background(background=img_desk, foreground=transformed)
    cv2.imshow('homography with normalization (2)', combined)
    cv2.waitKey(0)

    """
    Computing homography with RANSAC
    """
    print("2-3")
    match_list = get_match_list_with_threshold(des_desk, des_cover, threshold=0.8)
    N = 15
    srcP, destP = kp2p(src_kp=kp_cover, dest_kp=kp_desk, match_list=match_list, N=N)
    th = 3
    # fix seed
    np.random.seed(2022)
    H = compute_homography_ransac(srcP, destP, th=th)
    transformed = cv2.warpPerspective(img_cover, H, (img_desk.shape[1], img_desk.shape[0]))
    cv2.imshow('homography with RANSAC (1)', transformed)
    combined = combine_with_background(background=img_desk, foreground=transformed)
    cv2.imshow('homography with RANSAC (2)', combined)
    cv2.waitKey(0)

    # harry potter
    img_hp = cv2.imread("CV_Assignment_2_Images/hp_cover.jpg", cv2.IMREAD_GRAYSCALE)
    img_hp_resized = cv2.resize(img_hp, (img_cover.shape[1], img_cover.shape[0]), interpolation=cv2.INTER_CUBIC)
    transformed = cv2.warpPerspective(img_hp_resized, H, (img_desk.shape[1], img_desk.shape[0]))
    cv2.imshow('homography with RANSAC - Harry Potter (1)', transformed)
    combined = combine_with_background(background=img_desk, foreground=transformed)
    cv2.imshow('homography with RANSAC - Harry Potter (2)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    Image Stitching
    """
    print("2-5")
    img_left = cv2.imread("CV_Assignment_2_Images/diamondhead-10.png", cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread("CV_Assignment_2_Images/diamondhead-11.png", cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp_left = orb.detect(img_left, None)
    kp_left, des_left = orb.compute(img_left, kp_left)

    kp_right = orb.detect(img_right, None)
    kp_right, des_right = orb.compute(img_right, kp_right)

    match_list = get_match_list_with_threshold(des_left, des_right, threshold=0.8)
    N = 15
    srcP, destP = kp2p(src_kp=kp_right, dest_kp=kp_left, match_list=match_list, N=N)
    th = 0.8
    np.random.seed(2022)
    H = compute_homography_ransac(srcP, destP, th=th)

    transformed = cv2.warpPerspective(img_right, H, (img_left.shape[1] + int(destP[0][0] - srcP[0][0]), img_left.shape[0]))
    distinction = img_left.shape[1]
    img_left = np.concatenate([img_left, np.zeros((img_left.shape[0], int(destP[0][0] - srcP[0][0])))], axis=1)
    transformed_stitching = combine_with_background(background=transformed, foreground=img_left)
    cv2.imshow('image stitching without gradation blending', transformed_stitching)
    cv2.waitKey(0)

    gradation_range = 200
    for i in range(img_left.shape[0]):
        for t,j in enumerate(range(distinction-gradation_range, distinction)):
            transformed_stitching[i][j] = (t/gradation_range) * transformed[i][j] + (1-t/gradation_range) * transformed_stitching[i][j]

    cv2.imshow('image stitching with gradation blending', transformed_stitching)
    cv2.waitKey(0)
    cv2.destroyAllWindows()