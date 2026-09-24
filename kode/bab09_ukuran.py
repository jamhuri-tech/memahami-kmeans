"""Bab 9: silhouette, Calinski-Harabasz, dan Davies-Bouldin dari nol.

Ketiga fungsi dicetak di naskah (Listing 9.1 dan 9.2) dan dicocokkan
dengan sklearn.metrics pada keempat data.
"""
import numpy as np
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples)

from bab09_data import kmeans, semua_data


def silhouette(X, label):
    D = ((X[:, None] - X[None]) ** 2).sum(axis=2)
    D = np.sqrt(np.maximum(D, 0))
    K = label.max() + 1
    nk = np.bincount(label, minlength=K)
    i = np.arange(len(X))
    # jumlah jarak setiap titik ke setiap klaster
    S = np.stack([D[:, label == k].sum(axis=1)
                  for k in range(K)], axis=1)
    lain = nk[label] - 1              # anggota lain
    a = S[i, label] / np.maximum(lain, 1)
    S = S / nk
    S[i, label] = np.inf
    b = S.min(axis=1)
    s = (b - a) / np.maximum(a, b)
    s[lain == 0] = 0                  # klaster satu titik
    return s


def calinski_harabasz(X, label):
    K, n = label.max() + 1, len(X)
    mu = np.array([X[label == k].mean(axis=0)
                   for k in range(K)])
    nk = np.bincount(label)
    W = ((X - mu[label]) ** 2).sum()
    B = (nk * ((mu - X.mean(axis=0)) ** 2).sum(axis=1)).sum()
    return (B / (K - 1)) / (W / (n - K))


def davies_bouldin(X, label):
    K = label.max() + 1
    mu = np.array([X[label == k].mean(axis=0)
                   for k in range(K)])
    jarak = np.sqrt(((X - mu[label]) ** 2).sum(axis=1))
    s = np.bincount(label, weights=jarak) / np.bincount(label)
    M = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(axis=2))
    np.fill_diagonal(M, np.inf)
    R = (s[:, None] + s[None, :]) / M
    return R.max(axis=1).mean()


def angka(v, batas=1e-12):
    return f"{'< ' + format(batas, '.0e'):>11s}" if v < batas \
        else f"{v:11.1e}"


if __name__ == "__main__":
    print("data      K  |silhouette|  CH relatif  DB relatif")
    for nama, X, _ in semua_data():
        for K in (3, 6):
            lab = kmeans(X, K).labels_
            e1 = np.abs(silhouette(X, lab)
                        - silhouette_samples(X, lab)).max()
            ch = calinski_harabasz_score(X, lab)
            e2 = abs(calinski_harabasz(X, lab) - ch) / ch
            db = davies_bouldin_score(X, lab)
            e3 = abs(davies_bouldin(X, lab) - db) / db
            print(f"{nama:8s} {K:2d}  {angka(e1)}  {angka(e2)}"
                  f"  {angka(e3)}")
