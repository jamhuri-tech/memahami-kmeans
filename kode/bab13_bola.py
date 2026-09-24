"""Bab 13: spherical K-means untuk data yang klasternya berupa arah.

arah(d): tiga arah acak di R^d; panjang setiap titik tersebar
log-normal, sehingga panjang tidak membawa informasi klaster.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from bab04_data import BENIH


def arah(d, n_per=200, sebar=0.15, benih=BENIH):
    rng = np.random.default_rng(benih)
    U = rng.normal(size=(3, d))
    if d == 2:                        # tiga arah yang jelas terpisah
        U = np.array([[np.cos(a), np.sin(a)] for a in (0.3, 1.2, 2.1)])
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    X, y = [], []
    for k in range(3):
        V = U[k] + sebar * rng.normal(size=(n_per, d)) / np.sqrt(d) * 2
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        X.append(V * rng.lognormal(0.5, 0.8, size=(n_per, 1)))
        y += [k] * n_per
    return np.vstack(X), np.array(y)


def spherical_kmeans(X, C, maks_iter=100):
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    label = None
    for _ in range(maks_iter):
        baru = (Z @ C.T).argmax(axis=1)   # kosinus terbesar
        if label is not None and np.array_equal(baru, label):
            break
        label = baru
        C = np.array([Z[label == k].sum(axis=0)
                      for k in range(len(C))])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return label, C, (Z * C[label]).sum()


if __name__ == "__main__":
    print("   d   K-means  K-means+normal  spherical")
    for d in (2, 10, 50):
        X, y = arah(d)
        a = ari(y, KMeans(3, n_init=10, random_state=BENIH)
                .fit(X).labels_)
        Z = X / np.linalg.norm(X, axis=1, keepdims=True)
        b = ari(y, KMeans(3, n_init=10, random_state=BENIH)
                .fit(Z).labels_)
        hasil = max((spherical_kmeans(X, X[np.random.RandomState(s)
                     .choice(len(X), 3, replace=False)])
                     for s in range(10)), key=lambda h: h[2])
        print(f"{d:4d} {a:9.3f} {b:15.3f} {ari(y, hasil[0]):10.3f}")
