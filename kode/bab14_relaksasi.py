"""Bab 14: kernel K-means sebagai masalah nilai eigen yang dikekang.

Objektif kernel K-means = tr(K) - tr(H^T K H), dengan H matriks
indikator ternormalkan (H_ik = 1/sqrt(n_k)). Tanpa kekangan diskret,
maksimum tr(H^T K H) atas H ortonormal adalah jumlah K nilai eigen
terbesar matriks K.
"""
import numpy as np

from bab11_data import enam_data
from bab14_kernel import rbf, terbaik


def jejak(Km, label, K):
    H = np.eye(K)[label]
    H /= np.sqrt(H.sum(axis=0))
    return np.trace(H.T @ Km @ H)


if __name__ == "__main__":
    data = {n: (X, y) for n, X, y in enam_data()}
    print("data    gamma   batas eigen  label benar  kernel K-means")
    for nama in ("bulan", "cincin"):
        X, y = data[nama]
        for g in (5, 20):
            Km = rbf(X, g)
            batas = np.linalg.eigvalsh(Km)[-2:].sum()
            (lab, _), _ = terbaik(Km, 2)
            print(f"{nama:7s} {g:5g} {batas:13.2f} {jejak(Km, y, 2):12.2f}"
                  f" {jejak(Km, lab, 2):15.2f}")
