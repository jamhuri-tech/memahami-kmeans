"""Bab 7: mengapa scikit-learn memusatkan data sebelum menghitung jarak.

Data gumpalan digeser sejauh c di kedua koordinat. Label dari rumus
terurai ||c||^2 - 2 x.c dibandingkan dengan label dari selisih
langsung ||x - c||^2, dengan dan tanpa pemusatan.
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH, gumpalan

if __name__ == "__main__":
    X0, _ = gumpalan()
    C0 = KMeans(5, n_init=10, random_state=BENIH).fit(X0) \
        .cluster_centers_
    print("   geser   salah label: terurai  terpusat   KMeans")
    for c in (0.0, 1e4, 1e6, 1e7, 1e8, 1e9):
        X, C = X0 + c, C0 + c
        benar = ((X[:, None] - C[None]) ** 2).sum(axis=2).argmin(1)
        urai = ((C ** 2).sum(1) - 2 * X @ C.T).argmin(1)
        m = X.mean(axis=0)
        pusat = (((C - m) ** 2).sum(1)
                 - 2 * (X - m) @ (C - m).T).argmin(1)
        km = KMeans(5, init=C, n_init=1, max_iter=1).fit(X)
        print(f"  {c:6.0e}   {(urai != benar).sum():17d}"
              f"  {(pusat != benar).sum():8d}"
              f"  {(km.labels_ != benar).sum():7d}")
