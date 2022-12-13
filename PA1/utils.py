import numpy as np
import math
import cv2

def normalize(img):
    return img/255

def denormalize(img):
    return img*255

def nearest_pad(img, pad_H, pad_W):
    # assert pad_H >= 0 and pad_W >= 0, "Negative values for padding height or padding width occured"
    """
    img shape : (H,W)
    kernel shape : (kernel_H, kernel_W)
    pad_H = kernel_H -1 (even number)
    pad_W = kernel_W -1 (even number)
    output shape : (H+pad_H, W+pad_W) 
    """
    H,W = img.shape

    if pad_H == 0 and pad_W != 0:
        left = np.tile(img[:,0].reshape(-1,1), pad_W//2)
        right = np.tile(img[:,-1].reshape(-1,1), pad_W//2)
        ## concatenate => automatically copy inputs
        dest = np.concatenate((left, img, right), axis=1)

    elif pad_H != 0 and pad_W == 0:
        top = np.tile(img[0,:].reshape(1,-1), pad_H//2).reshape(pad_H//2,-1)
        bottom = np.tile(img[-1,:].reshape(1,-1), pad_H//2).reshape(pad_H//2,-1)
        dest = np.concatenate((top, img, bottom), axis=0)

    elif pad_H != 0 and pad_W != 0:
        left = np.tile(img[:,0].reshape(-1,1), pad_W//2)
        right = np.tile(img[:,-1].reshape(-1,1), pad_W//2)
        h_padded = np.concatenate((left, img, right), axis=1)

        top = np.tile(h_padded[0,:].reshape(1,-1), pad_H//2).reshape(pad_H//2,-1)
        bottom = np.tile(h_padded[-1,:].reshape(1,-1), pad_H//2).reshape(pad_H//2,-1)
        dest = np.concatenate((top, h_padded, bottom), axis=0)

    else:
        dest = img.copy()

    return dest

def gaussian(x, sigma):
    first_term = 1/(sigma * np.sqrt(2 * np.pi))
    second_term = np.power(np.e, -(x**2)/(2*(sigma**2)))

    return first_term * second_term

def quantize(dir):
    R,C = dir.shape
    quantized_dir = np.zeros_like(dir, dtype=int)
    bins = [i*45 for i in range(9)]
    bins[8] = 0
    for r in range(R):
        for c in range(C):
            quantized_dir[r][c] = bins[int((dir[r][c]+22.5)//45)]

    return quantized_dir

def bins_green(img, response_map, threshold):
    output = img.copy()
    H,W,C = img.shape

    for h in range(H):
        for w in range(W):
            if response_map[h][w] > threshold:
                output[h][w][0] = 0
                output[h][w][1] = 255
                output[h][w][2] = 0

    return output

def bins_green_circle(img, response_map):
    output = img.copy()
    H,W,C = img.shape

    for h in range(H):
        for w in range(W):
            if response_map[h][w] != 0:
                cv2.circle(output, (w,h), 3, (0,255,0),2)

    return output

