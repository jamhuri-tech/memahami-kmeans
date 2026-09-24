"""Bab 4: tiga identitas jumlah kuadrat, diperiksa dengan angka.

(1) sum ||x - m||^2 = sum ||x - xbar||^2 + n ||xbar - m||^2
(2) jumlah kuadrat klaster = (1/2n) jumlah kuadrat jarak antarpasangan
(3) total T = dalam W + antar B
"""
import numpy as np

from bab04_data import delapan_titik, gumpalan
from bab04_inersia import centroid, inersia


def jk_ke(X, m):
    return ((X - m) ** 2).sum()


def pasangan(X):
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    return D.sum() / (2 * len(X))


def dekomposisi(X, label, K):
    xbar = X.mean(axis=0)
    mu = centroid(X, label, K)
    nk = np.bincount(label, minlength=K)
    T = jk_ke(X, xbar)
    W = inersia(X, label, K)
    B = (nk * ((mu - xbar) ** 2).sum(axis=1)).sum()
    return T, W, B


if __name__ == "__main__":
    X = delapan_titik()
    P = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    C1 = X[P == 0]
    m = np.array([3.0, 2.0])
    xbar = C1.mean(axis=0)
    kiri = jk_ke(C1, m)
    kanan = jk_ke(C1, xbar) + len(C1) * ((xbar - m) ** 2).sum()
    print(f"(1) klaster ABCD, m = (3, 2): {kiri:.4f} = {kanan:.4f}")
    print(f"(2) klaster ABCD: {jk_ke(C1, xbar):.4f} = "
          f"{pasangan(C1):.4f}")
    T, W, B = dekomposisi(X, P, 2)
    print("(3) T = W + B")
    print(f"    delapan titik: T {T:.4f}  W {W:.4f}  B {B:.4f}")
    Xg, yg = gumpalan()
    T, W, B = dekomposisi(Xg, yg, 5)
    print(f"    gumpalan: T {T:.2f}  W {W:.2f}  B {B:.2f}")
    print(f"    gumpalan: selisih T - W - B = {abs(T - W - B):.1e}")
