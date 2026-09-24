"""Bab 12: K-means sebagai batas campuran Gaussian dengan sigma -> 0.

Soft K-means: kovarians sigma^2 I dan bobot sama, keduanya tetap;
hanya centroid yang ditaksir. Dari centroid awal yang sama dengan
algoritma Lloyd, centroid akhirnya dibandingkan dengan hasil Lloyd.
"""
import numpy as np
from scipy.special import softmax

from bab04_data import BENIH, gumpalan
from bab05_lloyd import jarak2, lloyd


def soft_kmeans(X, C, sigma, maks_iter=500, tol=1e-12):
    for t in range(maks_iter):
        g = softmax(-jarak2(X, C) / (2 * sigma ** 2), axis=1)
        C_baru = g.T @ X / g.sum(axis=0)[:, None]
        if np.abs(C_baru - C).max() < tol:
            break
        C = C_baru
    return C_baru, g, t + 1


if __name__ == "__main__":
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    C0 = X[rng.choice(len(X), 5, replace=False)]
    C_l, _, _ = lloyd(X, C0)
    print(" sigma   |mu soft - mu Lloyd|  rata max gamma  iterasi")
    for sigma in (3.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05):
        C, g, it = soft_kmeans(X, C0, sigma)
        print(f"{sigma:6.2f} {np.abs(C - C_l).max():20.2e}"
              f" {g.max(axis=1).mean():15.4f} {it:8d}")
