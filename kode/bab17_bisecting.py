"""Bab 17: bisecting K-means dari nol, lawan K-means dan
BisectingKMeans scikit-learn.

bisecting dicetak di naskah (Listing 17.2): klaster berinersia terbesar
dibelah dua dengan K-means (K = 2) sampai banyaknya klaster K.
"""
import numpy as np
from sklearn.cluster import BisectingKMeans, KMeans, kmeans_plusplus

from bab04_data import BENIH, gumpalan
from bab05_lloyd import lloyd
from bab08_awal import kisi


def inersia_label(X, label):
    return sum(((X[label == k] - X[label == k].mean(0)) ** 2).sum()
               for k in np.unique(label))


def haluskan(X, label):
    """Algoritma Lloyd penuh dari centroid hasil bisecting."""
    C0 = np.array([X[label == k].mean(axis=0)
                   for k in range(label.max() + 1)])
    return lloyd(X, C0)[1]


def bisecting(X, K, benih=0):
    label = np.zeros(len(X), dtype=int)
    pohon = []
    for baru in range(1, K):
        jk = [((X[label == j] - X[label == j].mean(0)) ** 2)
              .sum() for j in range(baru)]
        k = int(np.argmax(jk))          # inersia terbesar
        idx = np.flatnonzero(label == k)
        C0, _ = kmeans_plusplus(X[idx], 2,
                                random_state=benih + baru)
        _, lab2, _ = lloyd(X[idx], C0)
        label[idx[lab2 == 1]] = baru    # belahan kedua
        pohon.append((k, baru))
    return label, pohon


if __name__ == "__main__":
    print("data       cara              median J/J*   >1.1 J*")
    for nama, X, K in (("gumpalan", gumpalan()[0], 5),
                       ("kisi", kisi(), 25)):
        hasil = {
            "KMeans n_init=1": [KMeans(K, n_init=1, random_state=s)
                                .fit(X).inertia_ for s in range(20)],
            "KMeans n_init=10": [KMeans(K, n_init=10, random_state=s)
                                 .fit(X).inertia_ for s in range(20)],
            "bisecting kita": [inersia_label(X, bisecting(X, K, s)[0])
                               for s in range(20)],
            "bisecting+Lloyd": [inersia_label(X, haluskan(
                X, bisecting(X, K, s)[0])) for s in range(20)],
            "BisectingKMeans": [BisectingKMeans(K, random_state=s)
                                .fit(X).inertia_ for s in range(20)]}
        J_min = min(min(v) for v in hasil.values())
        for cara, J in hasil.items():
            J = np.array(J) / J_min
            print(f"{nama:10s} {cara:17s} {np.median(J):11.4f}"
                  f" {(J > 1.1).sum():8d}")
