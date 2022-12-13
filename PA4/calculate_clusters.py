import glob
import os
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering


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

if __name__ == "__main__":
    sift_feat_dir = os.path.join("./feats","*.sift")
    total_sift_arr = get_total_sift_arr(sift_feat_dir)

    n_clusters = 8
    ## kmeans
    do_kmeans = True
    if do_kmeans:
        kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=500).fit(total_sift_arr)
        np.save(f'kmeans_{n_clusters}.npy', kmeans_model.cluster_centers_)
        np.save(f"labels_{n_clusters}.npy", kmeans_model.predict(total_sift_arr))

