"""Bab 19: memilih K untuk fitur log RFM.

Untuk K = 2..10: inersia, silhouette, Calinski-Harabasz, Davies-Bouldin
(fungsi scikit-learn, sama dengan Bab 9), kestabilan subsampel dari
bab09_pilih, dan BIC campuran Gaussian kovarians penuh. Lalu gap
statistic dari bab09_gap, dan varians komponen campuran Gaussian pada
arah log F, yang menjelaskan mengapa BIC terus turun.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH
from bab09_gap import gap
from bab09_pilih import kestabilan
from bab19_data import baca, bersihkan, fitur_log, rfm

if __name__ == "__main__":
    tabel = rfm(*bersihkan(baca()))
    Z = fitur_log(tabel)
    print(" K   inersia  silh     CH     DB   stabil       BIC")
    for K in range(2, 11):
        km = KMeans(K, n_init=10, random_state=BENIH).fit(Z)
        gm = GaussianMixture(K, covariance_type="full", n_init=3,
                             random_state=BENIH).fit(Z)
        print(f"{K:2d}  {km.inertia_:8.1f} "
              f"{silhouette_score(Z, km.labels_):5.3f} "
              f"{calinski_harabasz_score(Z, km.labels_):6.1f} "
              f"{davies_bouldin_score(Z, km.labels_):6.3f} "
              f"{kestabilan(Z, K):8.3f} {gm.bic(Z):9.1f}")
    _, _, K_gap = gap(Z, 10)
    print(f"gap statistic (aturan satu galat baku): K = {K_gap}")

    print()
    print("campuran Gaussian K = 6, per komponen:")
    gm = GaussianMixture(6, covariance_type="full", n_init=3,
                         random_state=BENIH).fit(Z)
    rata, sb = np.log(tabel.F).mean(), np.log(tabel.F).std()
    urut = np.argsort(gm.means_[:, 1])
    print("  bobot  rata-rata F  varians arah log F")
    for k in urut:
        F = np.exp(gm.means_[k, 1] * sb + rata)
        print(f"  {gm.weights_[k]:5.3f}  {F:11.3f}  "
              f"{gm.covariances_[k][1, 1]:18.2e}")
    for f in (1, 2, 3):
        print(f"bagian pelanggan dengan F = {f}: "
              f"{(tabel.F == f).mean():.3f}")
