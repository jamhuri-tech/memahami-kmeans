"""Bab 15: Elkan dan Hamerly memberi label yang sama dengan Lloyd pada
setiap iterasi, dengan jauh lebih sedikit perhitungan jarak."""
import time

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import load_digits, make_blobs

from bab04_data import BENIH, gumpalan
from bab08_awal import kisi
from bab15_cepat import elkan, hamerly, lloyd_hitung


def data_uji():
    rng = np.random.default_rng(BENIH)
    Xb, _ = make_blobs(20000, centers=50, cluster_std=1.0,
                       center_box=(-40, 40), random_state=BENIH)
    return [("gumpalan", gumpalan()[0], 5), ("kisi", kisi(), 25),
            ("digits", load_digits().data, 10),
            ("blobs 20k", Xb, 50),
            ("seragam 20k", rng.uniform(0, 1, (20000, 2)), 50),
            ("acak d=50", rng.normal(size=(5000, 50)), 20)]


if __name__ == "__main__":
    print("data           K  iter  sama   Elkan  Hamerly  (bagian)")
    for nama, X, K in data_uji():
        C0, _ = kmeans_plusplus(X, K, random_state=0)
        a = lloyd_hitung(X, C0)
        e = elkan(X, C0)
        h = hamerly(X, C0)
        sama = all(len(r[2]) == len(a[2]) and all(
            np.array_equal(p, q) for p, q in zip(r[2], a[2]))
            for r in (e, h))
        print(f"{nama:12s} {K:3d} {len(a[2]):5d}  {str(sama):5s}"
              f" {sum(e[3]) / sum(a[3]):7.3f}"
              f" {sum(h[3]) / sum(a[3]):8.3f}")
