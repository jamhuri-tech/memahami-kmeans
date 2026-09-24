"""Bab 14: kernel K-means dari nol.

rbf dan kernel_kmeans dicetak di naskah (Listing 14.1). Jarak ke
centroid di ruang fitur dihitung dari matriks kernel saja, memakai
rumus inersia antarpasangan dari Bab 4.
"""
import numpy as np


def rbf(X, gamma):
    D = ((X[:, None] - X[None]) ** 2).sum(axis=2)
    return np.exp(-gamma * D)


def kernel_kmeans(Km, label, K, maks_iter=100):
    n = len(Km)
    for _ in range(maks_iter):
        H = np.zeros((n, K))
        H[np.arange(n), label] = 1
        nk = np.maximum(H.sum(axis=0), 1)
        # ||phi(x_i) - mu_k||^2 tanpa suku K_ii
        silang = Km @ H / nk               # rata-rata K_ij
        dalam = np.einsum("jk,jl,lk->k", H, Km, H) / nk ** 2
        D = -2 * silang + dalam
        baru = D.argmin(axis=1)
        if np.array_equal(baru, label):
            break
        label = baru
    J = (np.diag(Km) + D[np.arange(n), label]).sum()
    return label, J


def terbaik(Km, K, ulang=20, benih=0):
    rng = np.random.RandomState(benih)
    hasil = [kernel_kmeans(Km, rng.randint(K, size=len(Km)), K)
             for _ in range(ulang)]
    return min(hasil, key=lambda h: h[1]), hasil
