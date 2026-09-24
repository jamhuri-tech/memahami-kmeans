"""Bab 16: ukuran langkah 1/n_k lawan ukuran langkah tetap.

K-means online (satu titik per langkah) pada 20 000 titik gumpalan,
tiga lintasan. Dicetak inersia relatif terhadap Lloyd dan simpangan
centroid di akhir.
"""
import numpy as np
from sklearn.cluster import kmeans_plusplus

from bab04_data import BENIH, gumpalan
from bab05_lloyd import lloyd
from bab16_cocok import inersia


def online(X, C, urutan, eta=None):
    C = C.astype(float).copy()
    n_k = np.zeros(len(C))
    for i in urutan:
        k = ((C - X[i]) ** 2).sum(axis=1).argmin()
        n_k[k] += 1
        C[k] += (X[i] - C[k]) * (1 / n_k[k] if eta is None else eta)
    return C


if __name__ == "__main__":
    X, _ = gumpalan(20000)
    C0, _ = kmeans_plusplus(X, 5, random_state=0)
    C_l, _, _ = lloyd(X, C0)
    J_l = inersia(X, C_l)
    urutan = np.concatenate([np.random.default_rng(BENIH + p)
                             .permutation(len(X)) for p in range(3)])
    print("langkah      J / J_Lloyd - 1   |mu - mu_Lloyd| maks")
    for eta in (None, 0.001, 0.01, 0.1):
        C = online(X, C0, urutan, eta)
        urut = [np.argmin(((C_l - c) ** 2).sum(axis=1)) for c in C]
        print(f"{'1/n_k' if eta is None else eta:>8}  {inersia(X, C) / J_l - 1:16.1e}"
              f"  {np.abs(C - C_l[urut]).max():20.4f}")
