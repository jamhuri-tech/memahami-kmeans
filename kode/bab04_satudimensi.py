"""Bab 4: K-means satu dimensi yang tepat, dengan pemrograman dinamis.

kmeans_1d_tepat dicetak di naskah (Listing 4.3). Bagian utama
membandingkannya dengan KMeans scikit-learn berawal acak.
"""
from math import comb

import numpy as np
from sklearn.cluster import KMeans

from bab04_data import garis
from bab04_semua import stirling2


def kmeans_1d_tepat(x, K):
    x = np.sort(x)
    n = len(x)
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x ** 2)])

    def biaya(i, j):                  # jumlah kuadrat x[i:j]
        m = j - i
        return s2[j] - s2[i] - (s1[j] - s1[i]) ** 2 / m

    D = np.full((K + 1, n + 1), np.inf)
    awal = np.zeros((K + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):   # klaster = x[i:j]
                c = D[k - 1, i] + biaya(i, j)
                if c < D[k, j]:
                    D[k, j], awal[k, j] = c, i
    batas, j = [], n                      # telusuri balik
    for k in range(K, 0, -1):
        batas.append(awal[k, j])
        j = awal[k, j]
    return D[K, n], sorted(batas)[1:]


if __name__ == "__main__":
    x = garis()
    n, K = len(x), 4
    J, batas = kmeans_1d_tepat(x, K)
    ukuran = np.diff([0] + batas + [n])
    print(f"n = {n}, K = {K}: optimum J = {J:.4f}")
    print(f"ukuran klaster: {ukuran.tolist()}")
    print(f"partisi beruas-ruas C(n-1,K-1) = {comb(n - 1, K - 1)}")
    print(f"semua partisi S(n,K) = {float(stirling2(n, K)):.1e}")
    hasil = np.array([KMeans(K, init="random", n_init=1,
                             random_state=s).fit(x[:, None]).inertia_
                      for s in range(1000)])
    kena = np.isclose(hasil, J, rtol=1e-9)
    print(f"scikit-learn, 1000 awal acak: {kena.sum()} optimum")
    print(f"J terburuk {hasil.max():.2f} "
          f"({hasil.max() / J:.2f} x optimum)")
    for j, c in zip(*np.unique(hasil.round(4), return_counts=True)):
        print(f"  J = {j:8.4f}: {c:4d} kali")
