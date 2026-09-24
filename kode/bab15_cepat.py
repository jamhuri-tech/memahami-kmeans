"""Bab 15: algoritma Lloyd yang dipercepat dengan ketaksamaan segitiga.

lloyd_hitung : Lloyd biasa yang menghitung semua n x K jarak
elkan        : K batas bawah per titik (Listing 15.1)
hamerly      : satu batas bawah per titik (Listing 15.2)
Ketiganya mengembalikan (label, centroid, riwayat label, cacah jarak
per iterasi),
dan memakai jarak Euclid (bukan kuadrat) supaya ketaksamaan segitiga
berlaku.
"""
import numpy as np


def jarak(A, B):
    return np.sqrt(((A[:, None] - B[None]) ** 2).sum(axis=2))


def rata(X, label, K):
    return np.array([X[label == k].mean(axis=0) for k in range(K)])


def lloyd_hitung(X, C, maks_iter=300):
    n, K = len(X), len(C)
    riwayat, cacah, lama = [], [], None
    for _ in range(maks_iter):
        label = jarak(X, C).argmin(axis=1)
        cacah.append(n * K)
        riwayat.append(label)
        if lama is not None and np.array_equal(label, lama):
            break
        lama = label
        C = rata(X, label, K)
    return label, C, riwayat, cacah


def elkan(X, C, maks_iter=300):
    n, K = len(X), len(C)
    L = jarak(X, C)                    # batas bawah
    label = L.argmin(axis=1)
    u = L[np.arange(n), label]         # batas atas
    cacah, riwayat = [n * K], [label.copy()]
    for _ in range(maks_iter):
        C_baru = rata(X, label, K)     # langkah pembaruan
        geser = np.sqrt(((C_baru - C) ** 2).sum(axis=1))
        L = np.maximum(L - geser, 0)
        u += geser[label]
        C = C_baru
        CC = jarak(C, C)               # antarcentroid
        cacah.append(K * (K - 1) // 2)
        np.fill_diagonal(CC, np.inf)
        s = CC.min(axis=1) / 2
        calon = np.flatnonzero(u > s[label])
        u[calon] = np.sqrt(((X[calon] - C[label[calon]]) ** 2)
                           .sum(axis=1))   # kencangkan u
        cacah[-1] += len(calon)
        for k in range(K):
            lab = label[calon]
            m = calon[(lab != k) & (u[calon] > L[calon, k])
                      & (u[calon] > CC[lab, k] / 2)]
            d = np.sqrt(((X[m] - C[k]) ** 2).sum(axis=1))
            cacah[-1] += len(m)
            L[m, k] = d
            dekat = d < u[m]
            label[m[dekat]], u[m[dekat]] = k, d[dekat]
        riwayat.append(label.copy())
        if np.array_equal(riwayat[-1], riwayat[-2]):
            break
    return label, C, riwayat, cacah


def hamerly(X, C, maks_iter=300):
    n, K = len(X), len(C)
    D = jarak(X, C)
    urut = np.sort(D, axis=1)
    label = D.argmin(axis=1)
    u, l = urut[:, 0], urut[:, 1]      # terdekat, kedua
    cacah, riwayat = [n * K], [label.copy()]
    for _ in range(maks_iter):
        C_baru = rata(X, label, K)
        geser = np.sqrt(((C_baru - C) ** 2).sum(axis=1))
        i1, i2 = np.argsort(geser)[::-1][:2]
        u += geser[label]
        l -= np.where(label == i1, geser[i2], geser[i1])
        C = C_baru
        CC = jarak(C, C)
        cacah.append(K * (K - 1) // 2)
        np.fill_diagonal(CC, np.inf)
        batas = np.maximum(CC.min(axis=1)[label] / 2, l)
        calon = np.flatnonzero(u > batas)
        u[calon] = np.sqrt(((X[calon] - C[label[calon]]) ** 2)
                           .sum(axis=1))
        cacah[-1] += len(calon)
        calon = calon[u[calon] > batas[calon]]
        D = jarak(X[calon], C)         # semua jarak
        cacah[-1] += len(calon) * K
        urut = np.sort(D, axis=1)
        label[calon] = D.argmin(axis=1)
        u[calon], l[calon] = urut[:, 0], urut[:, 1]
        riwayat.append(label.copy())
        if np.array_equal(riwayat[-1], riwayat[-2]):
            break
    return label, C, riwayat, cacah
