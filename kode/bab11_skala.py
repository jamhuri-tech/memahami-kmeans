"""Bab 11: penskalaan peubah dapat memperbaiki atau merusak K-means.

(1) wine: 13 peubah dengan satuan berbeda, tiga kultivar.
(2) gumpalan + tiga peubah noise bersimpangan baku kecil.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import (MinMaxScaler, RobustScaler,
                                   StandardScaler)

from bab04_data import BENIH
from bab11_data import enam_data

SKALA = [("mentah", None), ("pembakuan", StandardScaler()),
         ("min-maks", MinMaxScaler()), ("robust", RobustScaler())]


def ari_skala(X, y, K):
    hasil = []
    for nama, s in SKALA:
        Z = X if s is None else s.fit_transform(X)
        lab = KMeans(K, n_init=10, random_state=BENIH).fit(Z).labels_
        hasil.append(adjusted_rand_score(y, lab))
    return hasil


if __name__ == "__main__":
    X, y = load_wine(return_X_y=True)
    nama = load_wine().feature_names
    sb = X.std(axis=0)
    urut = np.argsort(sb)[::-1]
    print("(1) wine, simpangan baku tiga peubah terbesar:")
    for j in urut[:3]:
        print(f"    {nama[j]:28s} {sb[j]:8.2f}")
    print(f"    bagian variansi total dari {nama[urut[0]]}: "
          f"{sb[urut[0]] ** 2 / (sb ** 2).sum():.4f}")
    print("    " + "  ".join(f"{n:>9s}" for n, _ in SKALA))
    print("ARI " + "  ".join(f"{v:9.3f}" for v in ari_skala(X, y, 3)))

    Xg, yg = enam_data()[0][1:]
    rng = np.random.default_rng(BENIH)
    print("(2) gumpalan + m peubah noise (simpangan baku 0.05):")
    print("  m " + "  ".join(f"{n:>9s}" for n, _ in SKALA))
    for m in (3, 10, 30):
        Xn = np.hstack([Xg, 0.05 * rng.normal(size=(len(Xg), m))])
        print(f"{m:3d} " + "  ".join(f"{round(v, 3) + 0.0:9.3f}"
                                     for v in ari_skala(Xn, yg, 3)))
    Xn = np.hstack([Xg, 0.05 * rng.normal(size=(len(Xg), 3))])
    print("    simpangan baku kolom sesudah penskalaan, m = 3")
    print("    (dua kolom berklaster, lalu tiga kolom noise):")
    for nama, s in SKALA[1:]:
        print(f"    {nama:9s} " + " ".join(
            f"{v:.2f}" for v in s.fit_transform(Xn).std(axis=0)))
