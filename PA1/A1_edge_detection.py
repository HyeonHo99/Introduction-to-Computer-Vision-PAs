import cv2
import math
import time
import os
from utils import *
from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d
import numpy as np
from numpy import arctan2,pi,sqrt,square,where


def compute_image_gradient(img):
    start_time = time.time()

    sobel_filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobel_filter_y = sobel_filter_x.T.copy()

    dx = cross_correlation_2d(img, sobel_filter_x)
    dy = cross_correlation_2d(img,sobel_filter_y)

    mag = sqrt(square(dx) + square(dy))
    dir = arctan2(-dy,dx) * (180/pi)
    dir[where(dir<0)] += 180

    print(f"\t\tGradient Computing Time:\t{time.time()-start_time}")

    return mag, dir

def non_maximum_suppression_dir(mag,dir):
    start_time = time.time()
    R,C = mag.shape
    suppressed_mag = mag.copy()
    d_r = [0,-1,-1,-1]
    d_c = [1,1,0,-1]

    ##quantize
    ## dir-dtype : float | quantized_dir-dtype : int
    quantized_dir = quantize(dir)

    for r in range(R):
        for c in range(C):
            idx = (quantized_dir[r][c] // 45) % 4
            n_r = r + d_r[idx]
            n_c = c + d_c[idx]
            if (0 <= n_r < R) and (0 <= n_c < C):
                if mag[r][c] < mag[n_r][n_c]:
                    suppressed_mag[r][c] = 0.0

            n_r = r - d_r[idx]
            n_c = c - d_c[idx]
            if (0 <= n_r < R) and (0 <= n_c < C):
                if mag[r][c] < mag[n_r][n_c]:
                    suppressed_mag[r][c] = 0.0


    print(f"\t\tNMS Time:\t{time.time() - start_time}")

    return suppressed_mag



if __name__ == "__main__":
    names = ["lenna", "shapes"]
    dirs = ["./A1_Images/lenna.png", "./A1_Images/shapes.png"]
    imgs = [normalize(cv2.imread(dir, cv2.IMREAD_GRAYSCALE)) for dir in dirs]
    magnitudes = imgs.copy()
    dirs = imgs.copy()

    if not os.path.exists("result"):
        os.makedirs("result")

    print("2-1")
    print("\tFirst apply gaussian filter (7,1.5) to images\n")
    for i in range(len(imgs)):
        gaussian_filter = get_gaussian_filter_2d(size=7, sigma=1.5)
        imgs[i] = cross_correlation_2d(imgs[i], gaussian_filter)
        # cv2.imshow(f"part_2_gaussian_filtered_{names[i]}", imgs[i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print("2-2")
    for i in range(len(imgs)):
        print(f"\t<{names[i]}>")
        magnitudes[i], dirs[i] = compute_image_gradient(imgs[i])
        cv2.imwrite(f"./result/part_2_edge_raw_{names[i]}.png", denormalize(magnitudes[i]))
        cv2.imshow(f"part_2_edge_raw_{names[i]}", magnitudes[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print()
    print("2-3")
    for i in range(len(imgs)):
        print(f"\t<{names[i]}>")
        suppressed_magnitude = non_maximum_suppression_dir(magnitudes[i], dirs[i])
        cv2.imwrite(f"./result/part_2_edge_sup_{names[i]}.png", denormalize(suppressed_magnitude))
        cv2.imshow(f"part_2_edge_sup_{names[i]}", suppressed_magnitude)
        cv2.waitKey(0)
        cv2.destroyAllWindows()