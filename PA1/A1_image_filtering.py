import cv2
import numpy as np
import math
import time
import os
from utils import *


def cross_correlation_1d(img, kernel):
    kernel_size = np.prod(kernel.shape, dtype=np.uint8)
    dest = img.copy()
    H,W = dest.shape

    ## vertical kernel
    if kernel.ndim == 2 and kernel.shape[0] > 1:
        padded = nearest_pad(img, pad_H=kernel_size-1, pad_W=0)
        for r in range(H):
            for c in range(W):
                dest[r][c] = np.sum(kernel.reshape(-1) * padded[r:r+kernel_size, c])

    ## horizontal kernel
    else:
        padded = nearest_pad(img, pad_H=0, pad_W=kernel_size-1)
        for r in range(H):
            for c in range(W):
                dest[r][c] = np.sum(kernel.reshape(-1) * padded[r, c:c+kernel_size])

    return dest

def cross_correlation_2d(img, kernel):
    kernel_H, kernel_W = kernel.shape
    ## src: nearest-padded image
    padded = nearest_pad(img, pad_H=kernel_H-1, pad_W=kernel_W-1)
    dest = img.copy()
    H,W = dest.shape

    for r in range(H):
        for c in range(W):
            dest[r][c] = np.sum(kernel*padded[r:r+kernel_H, c:c+kernel_W])
            # dest[r][c] = np.sum(np.multiply(kernel,padded[r:r+kernel_H, c:c+kernel_W]))

    return dest



def get_gaussian_filter_1d(size, sigma):
    # assert size%2 == 1, "Size of filter should be an odd number"

    L = size//2
    kernel = np.array([gaussian(i,sigma) for i in range(-L,L+1)])
    ## normalize
    kernel /= np.sum(kernel)

    return kernel

def get_gaussian_filter_2d(size, sigma):
    # assert size % 2 == 1, "Size of filter should be an odd number"

    kernel_1d = get_gaussian_filter_1d(size,sigma).reshape(1,-1)
    kernel_2d = np.matmul(kernel_1d.T, kernel_1d)
    ## normalize
    kernel_2d /= np.sum(kernel_2d)

    return kernel_2d


if __name__ == "__main__":
    names = ["lenna", "shapes"]
    dirs = ["./A1_Images/lenna.png","./A1_Images/shapes.png"]
    imgs = [normalize(cv2.imread(dir,cv2.IMREAD_GRAYSCALE)) for dir in dirs]

    if not os.path.exists("result"):
        os.makedirs("result")

    '''
    1-2.The Gaussian Filter
    c) Print the results of get_gaussian_filter_1d(5,1) and get_gaussian_filter_2d(5,1) 
    '''
    print("1-2")
    print("\tc)")
    print(f"\t\tget_gaussian_filter_1d(5,1)\n{get_gaussian_filter_1d(5,1)}\n")
    print(f"\t\tget_gaussian_filter_2d(5,2)\n{get_gaussian_filter_2d(5,1)}\n")


    '''
    d) Perform at least 9 different Gaussian filtering to an image
    '''
    kernel_sizes = [5, 11, 101]
    sigmas = [1, 6, 11]
    print("\td)")
    for i in range(len(imgs)):
        horizontal_grid = None
        vertical_grid = None

        for kernel_size in kernel_sizes:
            horizontal_grid = None

            for sigma in sigmas:
                gaussian_filter = get_gaussian_filter_2d(kernel_size, sigma)
                correlated = cross_correlation_2d(imgs[i], gaussian_filter)
                title = f"{kernel_size}x{kernel_size} s={sigma}"
                cv2.putText(correlated,title,(10,20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 1)

                if horizontal_grid is None:
                    horizontal_grid = correlated
                else:
                    horizontal_grid = np.concatenate((horizontal_grid, correlated), axis=1)

            if vertical_grid is None:
                vertical_grid = horizontal_grid
            else:
                vertical_grid = np.concatenate((vertical_grid, horizontal_grid), axis=0)

        cv2.imwrite(f"./result/part_1_gaussian_filtered_{names[i]}.png", denormalize(vertical_grid))
        cv2.imshow(f"part_1_gaussian_filtered_{names[i]}", vertical_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    '''
    e) Perform the Gaussian filtering by applying vertical and horizontal 1D kernels sequentially, and
    compare the result with a filtering with a 2D kernel.
    ==> size: 101x101, sigma: 6
    '''
    gaussian_size, gaussian_sigma = 101, 6
    print("\te)")
    for i in range(len(imgs)):
        print(f"\t\t<{names[i]}>")
        horizontal_kernel = get_gaussian_filter_1d(gaussian_size, gaussian_sigma)   ## shape: (101,)
        vertical_kernel = horizontal_kernel.copy().reshape(-1,1)                    ## shape: (101,1)
        square_kernel = get_gaussian_filter_2d(gaussian_size, gaussian_sigma)       ## shape: (101,101)

        ## 1d x two times
        start_time = time.time()
        v_filtered = cross_correlation_1d(imgs[i], vertical_kernel)
        h_filtered = cross_correlation_1d(v_filtered, horizontal_kernel)
        print(f"\t\t\t1D Gaussian two times:\t{time.time()-start_time}")

        ## 2d x one time
        start_time = time.time()
        filtered = cross_correlation_2d(imgs[i], square_kernel)
        print(f"\t\t\t2D Gaussian one time:\t{time.time() - start_time}")

        ## difference map, sum
        diff_map = h_filtered - filtered
        # cv2.imwrite(f"./result/part_1_difference_map_{names[i]}.png",denormalize(abs(diff_map)))
        cv2.imshow(f"part_1_difference_map_{names[i]}", abs(diff_map))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        diff_sum = np.sum(np.abs(diff_map))
        print(f"\t\t\tSum of absolute intensity differences:\t{diff_sum}\n")
