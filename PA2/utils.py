import numpy as np
from numpy import pi,sin,cos
import math
import cv2

def normalize(img):
    return img/255

def denormalize(img):
    return img*255

def keyboard2matrix(keyboard, M):

    # Move to the left by 5 pixels
    if keyboard == ord('a'):
        translate_matrix = np.identity(3)
        translate_matrix[0][2] = -1 * 5
        M = translate_matrix @ M

    # Move to the right by 5 pixels
    elif keyboard == ord('d'):
        translate_matrix = np.identity(3)
        translate_matrix[0][2] = 1 * 5
        M = translate_matrix @ M

    # Move to the upward by 5 pixels
    elif keyboard == ord('w'):
        translate_matrix = np.identity(3)
        translate_matrix[1][2] = -1 * 5
        M = translate_matrix @ M

    # Move to the downward by 5 pixels
    elif keyboard == ord('s'):
        translate_matrix = np.identity(3)
        translate_matrix[1][2] = 1 * 5
        M = translate_matrix @ M

    elif keyboard == ord('r'):
        rotate_matrix = np.identity(3)
        # angle = np.radians(-5)
        angle = (pi/180) * -5
        cosine = cos(angle)
        sine = sin(angle)
        rotate_matrix[0][0] = cosine
        rotate_matrix[0][1] = -1 * sine
        rotate_matrix[1][0] = sine
        rotate_matrix[1][1] = cosine
        M = rotate_matrix @ M

    elif keyboard == ord('t'):
        rotate_matrix = np.identity(3)
        # angle = np.radians(5)
        angle = (pi / 180) * 5
        cosine = cos(angle)
        sine = sin(angle)
        rotate_matrix[0][0] = cosine
        rotate_matrix[0][1] = -1 * sine
        rotate_matrix[1][0] = sine
        rotate_matrix[1][1] = cosine
        M = rotate_matrix @ M

    # Flip across y axis
    elif keyboard == ord('f'):
        flip_matrix = np.identity(3)
        flip_matrix[0][0] = -1
        M = flip_matrix @ M

    # Flip across x axis
    elif keyboard == ord('g'):
        flip_matrix = np.identity(3)
        flip_matrix[1][1] = -1
        M = flip_matrix @ M

    # Shrink the size by 5% along to x direction
    elif keyboard == ord('x'):
        scale_matrix = np.identity(3)
        scale_matrix[0][0] = 1 - 0.05
        M = scale_matrix @ M

    # Enlarge the size by 5% along to x direction
    elif keyboard == ord('c'):
        scale_matrix = np.identity(3)
        scale_matrix[0][0] = 1 + 0.05
        M = scale_matrix @ M

    # Shrink the size by 5% along to y direction
    elif keyboard == ord('y'):
        scale_matrix = np.identity(3)
        scale_matrix[1][1] = 1 - 0.05
        M = scale_matrix @ M

    # Enlarge the size by 5% along to y direction
    elif keyboard == ord('u'):
        scale_matrix = np.identity(3)
        scale_matrix[1][1] = 1 + 0.05
        M = scale_matrix @ M

    # Restore to the initial state
    elif keyboard == ord('h'):
        M = np.identity(3)

    return M

def get_match_list(des1_list, des2_list):
    # des1 : desk
    # des2 : cover
    match_list = []

    for i,des1 in enumerate(des1_list):
        distances = []
        for j,des2 in enumerate(des2_list):
            distance = [(x^y).bit_count() for x,y in zip(des1,des2)]
            distances.append(np.sum(distance))
        min_distance, j = np.min(distances), np.argmin(distances)
        match_list.append({'distance': min_distance, 'i': i, 'j': j})

    # sort according to distance
    match_list.sort(key=lambda match:match['distance'])

    return match_list


def get_dmatch_list(match_list):
    match_list = match_list[:10]
    dmatch_list = []
    for match in match_list:
        dmatch = cv2.DMatch()
        dmatch.queryIdx = match["i"]
        dmatch.trainIdx = match["j"]
        dmatch.distance = match["distance"].astype(float)
        dmatch_list.append(dmatch)

    return dmatch_list

def kp2p(src_kp, dest_kp, match_list, N=15):

    # desk (i) -> dest / cover (j) -> src
    list_src_idx = [match["j"] for match in match_list[:N]]
    list_dest_idx = [match["i"] for match in match_list[:N]]

    src_kp_np = np.array(src_kp)
    dest_kp_np = np.array(dest_kp)

    srcP = [kp.pt for kp in src_kp_np[list_src_idx[:]]]
    destP = [kp.pt for kp in dest_kp_np[list_dest_idx[:]]]

    srcP = np.array(srcP).reshape(N,2)
    destP = np.array(destP).reshape(N,2)

    return srcP, destP


def normalize_matrix(srcP):
    """
    input: either srcP (or destP)
    output: Ts (or Td)
    """

    ## Mean subtraction: translate the mean of the points to the origin (0, 0)
    mean_x, mean_y = np.mean(srcP, axis=0)
    mean_matrix = np.identity(3)
    mean_matrix[0][2] = - mean_x
    mean_matrix[1][2] = - mean_y

    ## Scaling: scale the points so that the longest distance to the origin is √2
    shifted = np.array([srcP[:,0] - mean_x, srcP[:,1] - mean_y])
    norm2_list = np.hypot(shifted[:,0], shifted[:,1])
    scale_factor = np.sqrt(2) / np.max(norm2_list)
    scale_matrix = np.identity(3)
    scale_matrix[0][0] = scale_factor
    scale_matrix[1][1] = scale_factor

    Ts = scale_matrix @ mean_matrix

    return Ts

def get_match_list_with_threshold(des1_list, des2_list, threshold=0.8):
    match_list = []

    for i, des1 in enumerate(des1_list):
        distances = []
        for j, des2 in enumerate(des2_list):
            distance = [(x ^ y).bit_count() for x, y in zip(des1, des2)]
            distances.append(np.sum(distance))

        distances_sorted = sorted(distances)
        ratio = distances_sorted[0] / distances_sorted[1]
        # check threshold
        if ratio < threshold:
            min_distance, j = np.min(distances), np.argmin(distances)
            match_list.append({'distance': min_distance, 'i': i, 'j': j})

    # sort according to distance
    match_list.sort(key=lambda match: match['distance'])

    return match_list


def combine_with_background(background, foreground):
    combined = background.copy()
    H = combined.shape[0]
    W = combined.shape[1]
    for i in range(H):
        for j in range(W):
            if foreground[i][j]:
                combined[i][j] = foreground[i][j]

    return combined