"""Bab 20: clustering hierarkis Ward sebagai pembanding dan awal K-means.

Pada data gumpalan (K = 5) dan data kisi (K = 25), inersia Ward, Ward
yang dilanjutkan algoritma Lloyd, dan KMeans dengan n_init=1 pada dua
puluh benih dibandingkan dengan inersia terbaik J* (KMeans n_init=50).
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from bab04_data import BENIH, gumpalan
from bab08_awal import kisi


def inersia(X, label):
    return sum(((X[label == k] - X[label == k].mean(axis=0)) ** 2).sum()
               for k in np.unique(label))


if __name__ == "__main__":
    print("data       Ward  Ward+Lloyd   KMeans n_init=1 (20 benih)")
    print("                                median    terburuk")
    for nama, X, K in (("gumpalan", gumpalan()[0], 5),
                       ("kisi", kisi(), 25)):
        J = KMeans(K, n_init=50, random_state=BENIH).fit(X).inertia_
        w = AgglomerativeClustering(K, linkage="ward").fit(X).labels_
        C = np.array([X[w == k].mean(axis=0) for k in range(K)])
        wl = KMeans(K, init=C, n_init=1).fit(X).inertia_
        satu = [KMeans(K, n_init=1, random_state=s).fit(X).inertia_
                for s in range(20)]
        print(f"{nama:9s} {inersia(X, w) / J:6.4f} {wl / J:10.4f}"
              f" {np.median(satu) / J:10.4f} {max(satu) / J:10.4f}")
