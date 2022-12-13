import os
import cv2
import numpy as np
import time
from utils import *


def compute_F_raw(M):
    N = M.shape[0]
    srcP = M[:,:2]
    destP = M[:,2:]

    A = []
    for i in range(N):
        x, y = srcP[i][0], srcP[i][1]
        _x, _y = destP[i][0], destP[i][1]
        # A.append(np.array([x*_x, x*_y, x, y*_x, y*_y, y, _x, _y, 1]))
        A.append(np.array([x * _x, _x * y, _x, x * _y, y * _y, _y, x, y, 1]))

    u, s, vh = np.linalg.svd(np.array(A))

    return vh[-1].reshape(3,3)


def compute_F_norm(M):
    N = M.shape[0]
    srcP = M[:, :2]
    destP = M[:, 2:]

    global img_H,img_W
    Ts = normalize_matrix(srcP, img_H, img_W)
    Td = normalize_matrix(destP, img_H, img_W)

    ## normalize srcP, destP
    srcP_3d = np.concatenate((srcP, np.ones((N, 1))), axis=1)
    destP_3d = np.concatenate((destP, np.ones((N, 1))), axis=1)
    normalized_srcP = np.array([(Ts @ row)[:2] for row in srcP_3d])
    normalized_destP = np.array([(Td @ row)[:2] for row in destP_3d])

    A = []
    for i in range(N):
        x, y = normalized_srcP[i][0], normalized_srcP[i][1]
        _x, _y = normalized_destP[i][0], normalized_destP[i][1]
        # A.append(np.array([x * _x, x * _y, x, y * _x, y * _y, y, _x, _y, 1]))
        A.append(np.array([x * _x, _x * y, _x, x*_y, y * _y, _y, x, y, 1]))


    u, s, vh = np.linalg.svd(np.array(A))
    F = vh[-1].reshape(3,3)

    U, S, VH = np.linalg.svd(F)
    s[-1] = 0
    F = U @ np.diag(S) @ VH
    denormalized = Ts.T @ F @ Td

    return denormalized

def compute_F_mine(M):
    start_time = time.time()

    N = M.shape[0]
    srcP = M[:, :2]
    destP = M[:, 2:]

    global img_H, img_W
    Ts = normalize_matrix_mine(srcP, img_H, img_W)
    Td = normalize_matrix_mine(destP, img_H, img_W)

    ## normalize srcP, destP
    srcP_3d = np.concatenate((srcP, np.ones((N, 1))), axis=1)
    destP_3d = np.concatenate((destP, np.ones((N, 1))), axis=1)
    normalized_srcP = np.array([(Ts @ row)[:2] for row in srcP_3d])
    normalized_destP = np.array([(Td @ row)[:2] for row in destP_3d])

    F = compute_F_norm_mine(N=N, Ts=Ts, Td=Td, normalized_srcP=normalized_srcP, normalized_destP=normalized_destP)
    error = compute_avg_reproj_error(M, F)
    # np.random.seed(2022)
    num = 64
    while True:
        random_indices = np.random.choice(N, num, replace=False)
        _F = compute_F_norm_mine(N=num, Ts=Ts, Td=Td, normalized_srcP=normalized_srcP[random_indices],
                                     normalized_destP=normalized_destP[random_indices])
        _error = compute_avg_reproj_error(M,_F)
        if _error < error:
            F = _F
            error = _error

        if time.time() - start_time >= 2.999:
            break

    # print(f"\tcompute_F_mine time:{time.time()-start_time}")

    return F


if __name__ == "__main__":

    print("Part 1")
    names = ["temple", "house", "library"]
    ftypes = ["png", "jpg", "jpg"]

    for i,name in enumerate(names):
        print(f"\t1-1 {name}")
        img1 = cv2.imread(f"{name}1.{ftypes[i]}")
        img2 = cv2.imread(f"{name}2.{ftypes[i]}")

        matches = np.loadtxt(f"{name}_matches.txt")

        # (b)
        raw_F = compute_F_raw(matches)

        global img_H, img_W
        img_H, img_W, img_C = img1.shape

        # (c)
        norm_F = compute_F_norm(matches)

        # (d)
        mine_F = compute_F_mine(matches)

        print(f"\t\tAverage Reprojection Errors ({name}1.{ftypes[i]} and {name}2.{ftypes[i]})")
        print(f"\t\t\tRaw = {compute_avg_reproj_error(matches, raw_F)}")
        print(f"\t\t\tNorm = {compute_avg_reproj_error(matches, norm_F)}")
        print(f"\t\t\tMine = {compute_avg_reproj_error(matches, mine_F)}")
        print()

    print("Visualization of epipolar lines")
    for i,name in enumerate(names):
        print(f"\t1-2 {name}")
        matches = np.loadtxt(f"{name}_matches.txt")
        img1 = cv2.imread(f"{name}1.{ftypes[i]}")
        img2 = cv2.imread(f"{name}2.{ftypes[i]}")

        # global img_H, img_W
        img_H, img_W, img_C = img1.shape
        F = compute_F_mine(matches)

        # np.random.seed(2022)
        random_indices = np.random.choice(len(matches), 3, replace=False)
        img1_points = [[int(x), int(y)] for x, y in matches[random_indices, :2]]
        img2_points = [[int(x), int(y)] for x, y in matches[random_indices, 2:]]
        draw_epipolar_lines(img1_points, img2_points, F, img1.copy(), img2.copy())

        while (True):
            keyboard = cv2.waitKey()
            if keyboard == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
               random_indices = np.random.choice(len(matches), 3, replace=False)
               img1_points = [[int(x), int(y)]for x,y in matches[random_indices, :2]]
               img2_points = [[int(x), int(y)]for x,y in matches[random_indices, 2:]]
               draw_epipolar_lines(img1_points, img2_points, F, img1.copy(), img2.copy())

