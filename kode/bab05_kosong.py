"""Bab 5: seri jarak dan klaster kosong.

(1) Seri: titik yang sama jauhnya dari dua centroid.
(2) Klaster kosong: centroid awal ketiga jauh dari semua titik.
"""
import warnings

import numpy as np
from sklearn.cluster import KMeans

from bab04_data import NAMA, delapan_titik
from bab05_lloyd import jarak2

if __name__ == "__main__":
    X = delapan_titik()
    C = np.array([[2.0, 1.0], [4.0, 1.0]])
    D = jarak2(X, C)
    print("seri: centroid (2, 1) dan (4, 1), titik C = (3, 1)")
    print(f"  d1 = {D[2, 0]:.2f}, d2 = {D[2, 1]:.2f}, "
          f"argmin memilih klaster {D[2].argmin() + 1}")

    C0 = np.array([[1.0, 0.0], [8.0, 5.0], [20.0, 20.0]])
    label = jarak2(X, C0).argmin(axis=1)
    print("klaster kosong: awal A, H, dan (20, 20)")
    print("  label:", (label + 1).tolist())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu3 = X[label == 2].mean(axis=0)
    print(f"  rata-rata klaster 3 yang kosong: {mu3.tolist()}")
    km = KMeans(3, init=C0, n_init=1).fit(X)
    print("  scikit-learn: label", (km.labels_ + 1).tolist())
    print(f"  centroid {km.cluster_centers_.round(2).tolist()}")
    print(f"  inertia_ {km.inertia_:.4f}, n_iter_ {km.n_iter_}")
    km1 = KMeans(3, init=C0, n_init=1, max_iter=1).fit(X)
    print(f"  sesudah 1 iterasi: {km1.cluster_centers_.round(2).tolist()}")
