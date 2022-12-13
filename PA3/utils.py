import numpy as np
import math
import cv2

def normalize(img):
    return img/255

def denormalize(img):
    return img*255

def normalize_matrix(srcP, img_H, img_W):
    """
    input: either srcP (or destP)
    output: Ts (or Td)
    """

    ## Mean subtraction
    mean_matrix = np.identity(3)
    mean_matrix[0][2] = - img_W/2
    mean_matrix[1][2] = - img_H/2

    ## Scaling
    scale_matrix = np.identity(3)
    scale_matrix[0][0] = 2/img_W
    scale_matrix[1][1] = 2/img_H

    Ts = scale_matrix @ mean_matrix

    return Ts

def normalize_matrix_mine(srcP, img_H, img_W):
    """
    input: either srcP (or destP)
    output: Ts (or Td)
    """

    ## Mean subtraction
    mean_matrix = np.identity(3)
    mean_matrix[0][2] = - img_W/2
    mean_matrix[1][2] = - img_H/2
    # mean_w = np.mean(srcP[0])
    # mean_h = np.mean(srcP[1])
    # mean_matrix[0][2] = - mean_w
    # mean_matrix[1][2] = - mean_h

    ## Scaling
    scale_matrix = np.identity(3)
    scale_matrix[0][0] = 2/img_W
    scale_matrix[1][1] = 2/img_H
    # sc_w = np.std(srcP[0])
    # sc_h = np.std(srcP[1])
    # scale_matrix[0][0] = 1 / sc_w
    # scale_matrix[1][1] = 1 / sc_h

    Ts = scale_matrix @ mean_matrix

    return Ts


def compute_F_norm_mine(N, Ts, Td, normalized_srcP, normalized_destP):
    A = []
    for i in range(N):
        x, y = normalized_srcP[i][0], normalized_srcP[i][1]
        _x, _y = normalized_destP[i][0], normalized_destP[i][1]
        # A.append(np.array([x * _x, x * _y, x, y * _x, y * _y, y, _x, _y, 1]))
        A.append(np.array([x * _x, _x * y, _x, x * _y, y * _y, _y, x, y, 1]))

    u, s, vh = np.linalg.svd(np.array(A))
    F = vh[-1].reshape(3,3)

    U, S, VH = np.linalg.svd(F)
    s[-1] = 0
    F = U @ np.diag(S) @ VH
    denormalized = Ts.T @ F @ Td

    return denormalized

def get_pt(W, line, first):
    if first:
        ret = (W, int(-(W*line[0]+line[2])/line[1]))
    else:
        ret = (0, int(-line[2]/line[1]))
    return ret

def draw_epipolar_lines(img1_points, img2_points, F, img1, img2):
    H,W,C = img1.shape
    rgb = [(255,0,0,), (0,255,0), (0,0,255)]

    for i, (img1_pt, img2_pt) in enumerate(zip(img1_points, img2_points)):
        cv2.circle(img1, (img1_pt[0], img1_pt[1]), 4, rgb[i], thickness=-1)
        cv2.circle(img2, (img2_pt[0], img2_pt[1]), 4, rgb[i], thickness=-1)

    lines1 = []
    for pt in img1_points:
        pt_ = F @ np.array([pt[0],pt[1],1]).T
        lines1.append(pt_)

    lines2 = []
    for pt in img2_points:
        pt_ = F @ np.array([pt[0], pt[1], 1]).T
        lines2.append(pt_)

    for i in range(3):
        cv2.line(img2, get_pt(W,lines1[i], first=True), get_pt(W,lines1[i],first=False), rgb[i], thickness=2)
        cv2.line(img1, get_pt(W,lines2[i], first=True), get_pt(W,lines2[i],first=False), rgb[i], thickness=2)

    combined = np.concatenate((img1,img2),axis=1)
    cv2.imshow("epipolar lines", combined)

def compute_avg_reproj_error(_M, _F):
    N = _M.shape[0]

    X = np.c_[ _M[:,0:2] , np.ones( (N,1) ) ].transpose()
    L = np.matmul( _F , X ).transpose()
    norms = np.sqrt( L[:,0]**2 + L[:,1]**2 )
    L = np.divide( L , np.kron( np.ones( (3,1) ) , norms ).transpose() )
    L = ( np.multiply( L , np.c_[ _M[:,2:4] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error = (np.fabs(L)).sum()

    X = np.c_[_M[:, 2:4], np.ones((N, 1))].transpose()
    L = np.matmul(_F.transpose(), X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = ( np.multiply( L , np.c_[ _M[:,0:2] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error += (np.fabs(L)).sum()

    return error/(N*2)