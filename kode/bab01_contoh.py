"""Bab 1: K-means pada data iris tanpa melihat spesiesnya.

KMeans dengan K = 3 dijalankan pada ukuran mahkota bunga (panjang dan
lebar petal), pada ukuran kelopak (sepal), dan pada keempatnya.
Spesies baru dipakai sesudahnya, untuk menghitung berapa bunga yang
masuk klaster yang didominasi spesiesnya sendiri.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from bab04_data import BENIH


def cocok(spesies, label):
    """Banyaknya titik yang spesiesnya sama dengan mayoritas klasternya."""
    return sum(np.bincount(spesies[label == k]).max()
               for k in np.unique(label))


if __name__ == "__main__":
    iris = load_iris()
    X, spesies = iris.data, iris.target
    print(f"{len(X)} bunga, 3 spesies, 50 bunga per spesies")
    print("peubah yang dipakai            cocok dengan spesies")
    for nama, kolom in (("mahkota (panjang, lebar)", [2, 3]),
                        ("kelopak (panjang, lebar)", [0, 1]),
                        ("keempatnya", [0, 1, 2, 3])):
        km = KMeans(3, n_init=10, random_state=BENIH).fit(X[:, kolom])
        print(f"{nama:30s} {cocok(spesies, km.labels_):4d} dari 150")

    print()
    km = KMeans(3, n_init=10, random_state=BENIH).fit(X[:, [2, 3]])
    urut = np.argsort(km.cluster_centers_[:, 0])   # klaster menurut
    label = np.argsort(urut)[km.labels_]           # panjang mahkota
    print("mahkota: klaster (kolom) lawan spesies (baris)")
    print("               klaster 1  klaster 2  klaster 3")
    for s, nama in enumerate(iris.target_names):
        n = np.bincount(label[spesies == s], minlength=3)
        print(f"{nama:12s}" + "".join(f"{v:11d}" for v in n))
    print("centroid (panjang, lebar mahkota, cm):")
    for k in range(3):
        c = km.cluster_centers_[urut[k]]
        print(f"  klaster {k + 1}: ({c[0]:.2f}, {c[1]:.2f})")
