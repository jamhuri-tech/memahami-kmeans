"""Bab 11: K-means dibandingkan dengan Ward dan DBSCAN.

DBSCAN: eps dipilih dari kisi 0.05..1.0 dengan ARI terbaik terhadap
label sebenarnya, min_samples = 5. Pilihan itu memakai label, jadi
angka DBSCAN di sini terlalu optimistis.
"""
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score as ari

from bab04_data import BENIH
from bab11_data import enam_data

if __name__ == "__main__":
    print("data       K-means   Ward  DBSCAN  eps")
    for nama, X, y in enam_data():
        K = y.max() + 1
        a = ari(y, KMeans(K, n_init=10, random_state=BENIH)
                .fit(X).labels_)
        b = ari(y, AgglomerativeClustering(K, linkage="ward")
                .fit(X).labels_)
        c, e = max((ari(y, DBSCAN(eps=e, min_samples=5)
                        .fit(X).labels_), e)
                   for e in np.arange(0.05, 1.01, 0.05))
        print(f"{nama:9s} {a:8.3f} {b:6.3f} {c:7.3f} {e:5.2f}")
