"""Bab 9: K yang dipilih setiap ukuran pada keempat data.

silhouette, CH : K = 2..10 yang memaksimumkan
DB             : K = 2..10 yang meminimumkan
gap            : aturan satu galat baku, K = 1..10
BIC            : campuran Gaussian sferis, K = 1..10, minimum
gap dan BIC dicetak ">=10" jika tidak ada pilihan di dalam rentang
kestabilan     : rata-rata ARI antara dua subsampel 80%, K = 2..10
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH
from bab09_data import kmeans, semua_data
from bab09_gap import gap
from bab09_ukuran import calinski_harabasz, davies_bouldin, silhouette


def kestabilan(X, K, ulang=10, benih=BENIH):
    rng = np.random.default_rng(benih)
    n, m = len(X), int(0.8 * len(X))
    ari = []
    for r in range(ulang):
        a, b = rng.choice(n, m, replace=False), \
            rng.choice(n, m, replace=False)
        ka, kb = kmeans(X[a], K, r), kmeans(X[b], K, r)
        irisan = np.intersect1d(a, b)
        ari.append(adjusted_rand_score(ka.predict(X[irisan]),
                                       kb.predict(X[irisan])))
    return np.mean(ari)


if __name__ == "__main__":
    rinci = []
    print("data      sebenarnya  silh  CH  DB  gap  BIC  stabil")
    for nama, X, K_benar in semua_data():
        Ks = range(2, 11)
        lab = {K: kmeans(X, K).labels_ for K in Ks}
        sil = [silhouette(X, lab[K]).mean() for K in Ks]
        ch = [calinski_harabasz(X, lab[K]) for K in Ks]
        db = [davies_bouldin(X, lab[K]) for K in Ks]
        _, _, K_gap = gap(X, 10)
        bic = [GaussianMixture(K, covariance_type="spherical",
                               n_init=3, random_state=BENIH)
               .fit(X).bic(X) for K in range(1, 11)]
        st = [kestabilan(X, K) for K in Ks]
        print(f"{nama:9s} {K_benar:10d} {2 + np.argmax(sil):5d}"
              f" {2 + np.argmax(ch):3d} {2 + np.argmin(db):3d}"
              f" {K_gap if K_gap else '>=10':>4}"
              f" {1 + np.argmin(bic) if np.argmin(bic) < 9 else '>=10':>4}"
              f" {2 + np.argmax(st):7d}")
        rinci.append((nama, sil, st))
    print("rincian K = 2..10:")
    for nama, sil, st in rinci:
        print(f"  {nama:8s} {'silh':6s}" + "".join(f"{v:5.2f}"
                                                   for v in sil))
        print(f"  {'':8s} {'stabil':6s}" + "".join(f"{v:5.2f}"
                                                  for v in st))
