"""Bab 17: K-means dengan klaster berukuran sama.

seimbang dicetak di naskah (Listing 17.3). Langkah penugasan menjadi
masalah penugasan: setiap centroid disalin sebanyak kapasitasnya
(n/K, dibulatkan), lalu setiap titik
dipasangkan dengan satu salinan dengan biaya kuadrat jarak terkecil
(algoritma Hungaria).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from bab04_data import BENIH
from bab11_data import enam_data


def seimbang(X, C, maks_iter=100):
    n, K = len(X), len(C)
    # kapasitas n//K, plus satu untuk n % K klaster pertama
    kapasitas = n // K + (np.arange(K) < n % K)
    pemilik = np.repeat(np.arange(K), kapasitas)   # panjang n
    label = None
    for _ in range(maks_iter):
        D = ((X[:, None] - C[None]) ** 2).sum(axis=2)
        _, kolom = linear_sum_assignment(D[:, pemilik])
        baru = pemilik[kolom]
        if label is not None and np.array_equal(baru, label):
            break
        label = baru
        C = np.array([X[label == k].mean(axis=0)
                      for k in range(K)])
    J = ((X - C[label]) ** 2).sum()
    return label, C, J


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    data = [("seragam", rng.uniform(0, 10, (600, 2)), 6),
            ("timpang", enam_data()[2][1], 3)]
    print("data      cara        J       ukuran klaster")
    for nama, X, K in data:
        km = KMeans(K, n_init=10, random_state=BENIH).fit(X)
        lab, C, J = seimbang(X, km.cluster_centers_)
        for cara, l, j in (("K-means", km.labels_, km.inertia_),
                           ("seimbang", lab, J)):
            print(f"{nama:9s} {cara:9s} {j:8.1f}   "
                  f"{sorted(np.bincount(l).tolist())}")
