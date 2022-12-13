import cv2
import numpy as np
import time
import os
from utils import *
from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d
import numpy as np


def compute_corner_response(img):
    start_time = time.time()

    R = np.zeros_like(img)
    eigvals = np.linalg.eigvals
    window_size = 5
    half_size = window_size//2
    kappa = 0.04

    sobel_filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobel_filter_y = sobel_filter_x.T.copy()

    ## gradient map
    dx = cross_correlation_2d(img, sobel_filter_x)
    dy = cross_correlation_2d(img, sobel_filter_y)
    cov_matrix = np.array([[0.0,0.0],
                           [0.0,0.0]])

    for r in range(R.shape[0] - window_size):
        for c in range(R.shape[1] - window_size):
            Ix_patch = dx[r:r+window_size,c:c+window_size]
            Iy_patch = dy[r:r+window_size,c:c+window_size]

            ## calculate Response
            cov_matrix[0][0] = np.sum(np.square(Ix_patch))
            cov_matrix[0][1] = cov_matrix[1][0] = np.sum(np.multiply(Ix_patch,Iy_patch))
            cov_matrix[1][1] = np.sum(np.square(Iy_patch))

            eigen_values = eigvals(cov_matrix)

            response = eigen_values[0]*eigen_values[1] - kappa*(eigen_values[0]+eigen_values[1])**2

            if response < 0:
                response = 0.0
            R[r+half_size][c+half_size] = response

    ## MinMax Normalize
    R /= np.max(R)

    print(f"\t\tCorner Response Computing Time:\t{time.time() - start_time}")
    return R

def non_maximum_suppression_win(R, winsize):
    start_time = time.time()

    suppressed_R = np.zeros_like(R)
    H,W = R.shape

    for h in range(winsize//2, H-winsize//2):
        for w in range(winsize//2, W-winsize//2):
            response = R[h][w]

            if response > 0.1:
                patch = R[h-winsize//2:h+winsize//2+1, w-winsize//2:w+winsize//2+1]
                if response == np.max(patch):
                    suppressed_R[h][w] = response

    end_time = time.time()
    print(f"\t\tCorner Response NMS Time:\t{end_time-start_time}")
    return suppressed_R

if __name__ == "__main__":
    names = ["lenna", "shapes"]
    dirs = ["./A1_Images/lenna.png", "./A1_Images/shapes.png"]
    imgs = [normalize(cv2.imread(dir, cv2.IMREAD_GRAYSCALE)) for dir in dirs]
    responses = [None, None]
    rgbs = [None, None]
    bins = [None, None]

    if not os.path.exists("result"):
        os.makedirs("result")

    print("3-1")
    print("\tFirst apply gaussian filter (7,1.5) to images\n")
    for i in range(len(imgs)):
        gaussian_filter = get_gaussian_filter_2d(size=7, sigma=1.5)
        imgs[i] = cross_correlation_2d(imgs[i], gaussian_filter)
        # cv2.imshow(f"part_3_gaussian_filtered_{names[i]}", imgs[i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print("3-2")
    for i in range(len(imgs)):
        print(f"\t<{names[i]}>")
        responses[i] = compute_corner_response(imgs[i])
        cv2.imwrite(f"./result/part_3_corner_raw_{names[i]}.png", denormalize(responses[i]))
        cv2.imshow(f"part_3_corner_raw_{names[i]}", responses[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n3-3")
    for i in range(len(imgs)):
        print(f"\t<{names[i]}>")
        rgbs[i] = cv2.cvtColor(cv2.imread(dirs[i], cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
        bins[i] = bins_green(rgbs[i], responses[i], threshold=0.1)
        cv2.imwrite(f"./result/part_3_corner_bin_{names[i]}.png", bins[i])
        cv2.imshow(f"part_3_corner_bin_{names[i]}", bins[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        suppressed_R = non_maximum_suppression_win(responses[i], winsize=11)
        suppressed_R_bins = bins_green_circle(rgbs[i], suppressed_R)
        cv2.imwrite(f"./result/part_3_corner_sup_{names[i]}.png", suppressed_R_bins)
        cv2.imshow(f"part_3_corner_sup_{names[i]}", suppressed_R_bins)
        cv2.waitKey(0)
        cv2.destroyAllWindows()