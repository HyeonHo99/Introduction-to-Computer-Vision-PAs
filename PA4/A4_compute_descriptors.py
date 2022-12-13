import os
import glob
import numpy as np


def get_sift_arr(directory):
    sift_arr = []
    for d in sorted(glob.glob(directory)):
        with open(d, "rb") as f:
            arr = np.array(np.fromfile(f, np.uint8).reshape(-1, 128)).astype(np.float32)
            sift_arr.append(arr)

    return sift_arr


def get_total_sift_arr(directory):
    total_sift_arr = []
    for d in sorted(glob.glob(directory)):
        with open(d, "rb") as f:
            arr = np.array(np.fromfile(f, np.uint8).reshape(-1, 128)).astype(np.float32)
            for row in arr:
                total_sift_arr.append(np.array(row))

    return np.array(total_sift_arr)


def norm_mine(residual_sum):
    eps = 1e-6
    residual_normalized = residual_sum / (np.cbrt(residual_sum @ residual_sum) + eps)
    residual_signs = np.sign(residual_normalized)
    residual_norm = residual_signs * np.log1p(residual_normalized * residual_signs)

    return residual_norm

if __name__ == "__main__":
    n_clusters = 8
    n_images = 1000
    dim_des = 1024
    sift_feat_dir = os.path.join("./feats", "*.sift")
    sift_arr = get_sift_arr(sift_feat_dir)
    total_sift_arr = get_total_sift_arr(sift_feat_dir)
    # print(f"sift_arr len: {len(sift_arr)}")
    # print(f"total_sift_arr shape: {total_sift_arr.shape}\n")

    ## pre-computed centers & labels
    centers = np.load(f'kmeans_{n_clusters}.npy')
    labels = np.load(f'labels_{n_clusters}.npy')

    start_end_list = []
    acc = 0
    for i in range(n_images):
        start_end_list.append([acc, acc + len(sift_arr[i])])
        acc += len(sift_arr[i])

    # print(f"centers shape: {centers.shape}")
    # print(f"labels shape: {labels.shape}")

    ret = []
    for i in range(n_images):
        start = start_end_list[i][0]
        end = start_end_list[i][1]
        cur_sift = sift_arr[i]
        cur_label = labels[start:end]
        ret_row = np.zeros(dim_des, dtype=float)
        for idx,cur_center in enumerate(centers):
            cur_sift_nearest = cur_sift[np.where(idx == cur_label)]
            residual = cur_sift_nearest - cur_center
            residual_sum = np.sum(residual, axis=0)
            residual_norm = norm_mine(residual_sum)
            idx *= 128
            _idx = idx + 128
            ret_row[idx:_idx] = residual_norm
        ret_row /= np.linalg.norm(ret_row, ord=4)
        ret.append(ret_row)

    # A4_2017314854.des
    with open("./A4_2017314854.des", 'wb') as f:
        f.write(np.array(n_images).astype('int32').tobytes())
        f.write(np.array(dim_des).astype('int32').tobytes())
        f.write(np.array(ret).astype('float32').tobytes())
