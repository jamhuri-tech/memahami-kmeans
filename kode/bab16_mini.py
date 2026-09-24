"""Bab 16: K-means online (MacQueen) dan mini-batch K-means.

macqueen dicetak di naskah (Listing 16.1), minibatch (Listing 16.2).
Keduanya memakai ukuran langkah 1/n_k per centroid, sehingga setiap
centroid selalu rata-rata semua titik yang pernah diberikan kepadanya.
"""
import numpy as np


def macqueen(X, C, urutan):
    C = C.astype(float).copy()
    n_k = np.zeros(len(C))
    for i in urutan:
        k = ((C - X[i]) ** 2).sum(axis=1).argmin()
        n_k[k] += 1
        C[k] += (X[i] - C[k]) / n_k[k]        # langkah 1/n_k
    return C


def minibatch(X, C, b, langkah, rng):
    C = C.astype(float).copy()
    n_k = np.zeros(len(C))
    for _ in range(langkah):
        B = X[rng.randint(0, len(X), b)]
        D = ((B[:, None] - C[None]) ** 2).sum(axis=2)
        lab = D.argmin(axis=1)
        for k in np.unique(lab):
            anggota = B[lab == k]
            C[k] = (C[k] * n_k[k] + anggota.sum(axis=0)) \
                / (n_k[k] + len(anggota))
            n_k[k] += len(anggota)
    return C
