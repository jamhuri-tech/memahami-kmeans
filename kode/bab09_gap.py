"""Bab 9: gap statistic (Tibshirani, Walther, dan Hastie 2001).

Fungsi gap dicetak di naskah (Listing 9.3). Data acuan: titik seragam
di kotak pembatas data, B kali.
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH


def log_W(X, K):
    km = KMeans(K, n_init=10, random_state=BENIH).fit(X)
    return np.log(km.inertia_)


def gap(X, K_maks, B=10, benih=BENIH):
    rng = np.random.default_rng(benih)
    bawah, atas = X.min(axis=0), X.max(axis=0)
    acuan = [rng.uniform(bawah, atas, size=X.shape)
             for _ in range(B)]
    G, s = [], []
    for K in range(1, K_maks + 1):
        lw = np.array([log_W(Xa, K) for Xa in acuan])
        G.append(lw.mean() - log_W(X, K))
        s.append(lw.std() * np.sqrt(1 + 1 / B))
    G, s = np.array(G), np.array(s)
    # K terkecil dengan Gap(K) >= Gap(K+1) - s(K+1);
    # None jika aturan itu tidak terpenuhi sampai K_maks
    ok = G[:-1] >= G[1:] - s[1:]
    K_pilih = int(np.argmax(ok)) + 1 if ok.any() else None
    return G, s, K_pilih
