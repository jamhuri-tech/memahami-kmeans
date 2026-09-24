"""Bab 13: tiga varian K-means yang kokoh terhadap pencilan.

kmedians       : jarak L1, median per koordinat (Listing 13.1)
kmedoids       : centroid berupa titik data, ketakmiripan apa pun,
                 cara bergantian lalu pertukaran PAM (Listing 13.2)
trimmed_kmeans : sebagian alpha titik terjauh diabaikan (Listing 13.3)
Setiap fungsi mengembalikan (label, pusat, objektif).
"""
import numpy as np


def kmedians(X, C, maks_iter=100):
    label = None
    for _ in range(maks_iter):
        D = np.abs(X[:, None] - C[None]).sum(axis=2)  # L1
        baru = D.argmin(axis=1)
        if label is not None and np.array_equal(baru, label):
            break
        label = baru
        C = np.array([np.median(X[label == k], axis=0)
                      for k in range(len(C))])
    return label, C, D.min(axis=1).sum()


def kmedoids(D, medoid, maks_iter=100):
    medoid = np.array(medoid)
    for _ in range(maks_iter):          # cara bergantian
        label = D[:, medoid].argmin(axis=1)
        baru = medoid.copy()
        for k in range(len(medoid)):
            a = np.flatnonzero(label == k)
            baru[k] = a[D[np.ix_(a, a)].sum(axis=0).argmin()]
        if np.array_equal(baru, medoid):
            break
        medoid = baru
    while True:                               # pertukaran PAM
        kini = D[:, medoid].min(axis=1).sum()
        terbaik = (kini, None, None)
        for j in range(len(medoid)):
            dasar = D[:, np.delete(medoid, j)].min(axis=1)
            biaya = np.minimum(dasar[:, None], D).sum(axis=0)
            o = biaya.argmin()
            if biaya[o] < terbaik[0] - 1e-9:
                terbaik = (biaya[o], j, o)
        if terbaik[1] is None:
            break
        medoid[terbaik[1]] = terbaik[2]
    label = D[:, medoid].argmin(axis=1)
    return label, medoid, D[:, medoid].min(axis=1).sum()


def trimmed_kmeans(X, C, alpha, maks_iter=100):
    n_simpan = len(X) - int(alpha * len(X))
    lama = None
    for _ in range(maks_iter):
        D = ((X[:, None] - C[None]) ** 2).sum(axis=2)
        label, d = D.argmin(axis=1), D.min(axis=1)
        buang = np.ones(len(X), dtype=bool)
        buang[np.argsort(d)[:n_simpan]] = False  # terdekat
        kunci = (label * 2 + buang).tobytes()
        if kunci == lama:
            break
        lama = kunci
        C = np.array([X[~buang & (label == k)].mean(axis=0)
                      for k in range(len(C))])
    return label, C, d[~buang].sum(), buang
