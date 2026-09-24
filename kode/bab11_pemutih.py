"""Bab 11: PCA dan pemutihan (whitening) sebelum K-means.

Data lonjong: pemutihan memakai kovarians gabungan semua titik, atau
kovarians dalam klaster (rata-rata kovarians per kelas, yang pada
data nyata tidak kita ketahui).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from bab04_data import BENIH
from bab11_data import enam_data


def putihkan(X, S):
    nilai, vektor = np.linalg.eigh(S)
    return (X - X.mean(axis=0)) @ vektor / np.sqrt(nilai)


def ari_km(X, y):
    lab = KMeans(y.max() + 1, n_init=10,
                 random_state=BENIH).fit(X).labels_
    return adjusted_rand_score(y, lab)


if __name__ == "__main__":
    _, X, y = enam_data()[1]
    S_total = np.cov(X, rowvar=False)
    S_dalam = sum(np.cov(X[y == k], rowvar=False) for k in range(3)) / 3
    print("lonjong, ARI K-means:")
    print(f"  mentah                         {ari_km(X, y):.3f}")
    print(f"  pemutihan kovarians gabungan   "
          f"{ari_km(putihkan(X, S_total), y):.3f}")
    print(f"  pemutihan kovarians dalam      "
          f"{ari_km(putihkan(X, S_dalam), y):.3f}")
