"""Bab 4: centroid dan inersia sebuah partisi.

Fungsi centroid dan inersia dicetak di naskah (Listing 4.1). Bagian
utama menghitung dua partisi delapan titik dan mencocokkan inersianya
dengan KMeans scikit-learn.
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import NAMA, delapan_titik


def centroid(X, label, K):
    return np.array([X[label == k].mean(axis=0)
                     for k in range(K)])


def inersia(X, label, K):
    mu = centroid(X, label, K)
    return ((X - mu[label]) ** 2).sum()


def tampilkan(X, label, K):
    mu = centroid(X, label, K)
    for k in range(K):
        anggota = "".join(NAMA[i] for i in np.flatnonzero(label == k))
        jk = ((X[label == k] - mu[k]) ** 2).sum()
        print(f"  C{k + 1} = {{{','.join(anggota)}}}  "
              f"mu = ({mu[k, 0]:.2f}, {mu[k, 1]:.2f})  J_k = {jk:.2f}")
    print(f"  J = {inersia(X, label, K):.4f}")


if __name__ == "__main__":
    X = delapan_titik()
    P = np.array([0, 0, 0, 0, 1, 1, 1, 1])      # ABCD | EFGH
    Q = np.array([0, 0, 0, 0, 0, 0, 0, 1])      # ABCDEFG | H
    print("partisi P:")
    tampilkan(X, P, 2)
    print("partisi Q:")
    tampilkan(X, Q, 2)
    km = KMeans(2, init=centroid(X, P, 2), n_init=1).fit(X)
    print(f"scikit-learn: inertia_ = {km.inertia_:.4f}, "
          f"score = {km.score(X):.4f}")
    X2 = np.vstack([X, X])                      # setiap titik dua kali
    km2 = KMeans(2, init=centroid(X, P, 2), n_init=1).fit(X2)
    print(f"data digandakan: inertia_ = {km2.inertia_:.4f}")
