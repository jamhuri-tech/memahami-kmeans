"""Bab 20: coreset ringan (Bachem, Lucic, dan Krause 2018).

coreset dicetak di naskah (Listing 20.1). Data: 198.500 titik, sembilan
gumpalan besar dan satu gumpalan kecil (500 titik) yang jauh; K = 10.
KMeans dijalankan pada m titik, dari sampel seragam atau dari coreset
berbobot, lalu inersia centroidnya diukur pada seluruh data dan
dibagi inersia KMeans pada seluruh data. Dua puluh benih.
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH


def data_coreset(benih=BENIH):
    rng = np.random.default_rng(benih)
    pusat = rng.uniform(-10, 10, (9, 2))
    besar = [c + rng.normal(size=(22_000, 2)) for c in pusat]
    kecil = np.array([40.0, 40.0]) + rng.normal(size=(500, 2))
    return np.vstack(besar + [kecil])


def coreset(X, m, rng):
    d2 = ((X - X.mean(axis=0)) ** 2).sum(axis=1)
    q = 0.5 / len(X) + 0.5 * d2 / d2.sum()  # separuh seragam
    i = rng.choice(len(X), m, p=q)
    return X[i], 1 / (m * q[i])     # titik dan bobotnya


def inersia(X, C):
    return ((X[:, None] - C[None]) ** 2).sum(axis=2).min(axis=1).sum()


if __name__ == "__main__":
    X = data_coreset()
    K = 10
    J = KMeans(K, n_init=3, random_state=BENIH).fit(X).inertia_
    print(f"{len(X)} titik, K = {K}; J/J* pada seluruh data")
    print("       ---- seragam -----  ----- coreset -----")
    print("    m    median  terburuk     median  terburuk")
    for m in (100, 300, 1000, 3000):
        rs, rc = [], []
        for s in range(20):
            rng = np.random.default_rng(s)
            S = X[rng.choice(len(X), m, replace=False)]
            km = KMeans(K, n_init=3, random_state=s).fit(S)
            rs.append(inersia(X, km.cluster_centers_) / J)
            S, w = coreset(X, m, rng)
            km = KMeans(K, n_init=3, random_state=s)
            km.fit(S, sample_weight=w)
            rc.append(inersia(X, km.cluster_centers_) / J)
        print(f"{m:5d}  {np.median(rs):8.3f} {max(rs):9.3f}"
              f"  {np.median(rc):9.3f} {max(rc):9.3f}")
