"""Bab 15: KMeans(algorithm="elkan") lawan algorithm="lloyd" pada
scikit-learn, dari centroid awal yang sama, dan waktu keduanya."""
import time

import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from bab15_cocok import data_uji

if __name__ == "__main__":
    print("data          label sama  n_iter lloyd  n_iter elkan")
    for nama, X, K in data_uji():
        C0, _ = kmeans_plusplus(X, K, random_state=0)
        a = KMeans(K, init=C0, n_init=1, tol=0,
                   algorithm="lloyd").fit(X)
        b = KMeans(K, init=C0, n_init=1, tol=0,
                   algorithm="elkan").fit(X)
        print(f"{nama:12s} {str(np.array_equal(a.labels_, b.labels_)):>11s}"
              f" {a.n_iter_:13d} {b.n_iter_:13d}")
