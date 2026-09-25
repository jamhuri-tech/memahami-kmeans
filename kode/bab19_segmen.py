"""Bab 19: empat segmen K-means pada fitur log RFM.

Segmen diurutkan menurut median M. Untuk setiap segmen dicetak
banyaknya pelanggan, median R, F, M, bagian dari seluruh belanja,
dan centroid yang dikembalikan ke satuan asal. Lalu kesesuaian dengan
campuran Gaussian dan dengan K-means pada penskalaan lain (ARI).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH
from bab19_data import baca, bersihkan, fitur_log, rfm


def segmen(tabel, K=4):
    """Label K-means pada fitur log, dinomori menurut median M."""
    km = KMeans(K, n_init=10, random_state=BENIH).fit(fitur_log(tabel))
    med = [tabel.M[km.labels_ == k].median() for k in range(K)]
    urut = np.argsort(np.argsort(med))
    return urut[km.labels_], km.cluster_centers_[np.argsort(med)]


if __name__ == "__main__":
    tabel = rfm(*bersihkan(baca()))
    label, C = segmen(tabel)
    L = np.column_stack([np.log1p(tabel.R), np.log(tabel.F),
                         np.log(tabel.M)])
    asal = np.exp(C * L.std(0) + L.mean(0))
    asal[:, 0] -= 1                      # balikan log(1 + R)
    total = tabel.M.sum()
    print("segmen  pelanggan  med R  med F  med M  bagian belanja")
    for k in range(4):
        t = tabel[label == k]
        print(f"{k + 1:6d} {len(t):10d} {t.R.median():6.0f} "
              f"{t.F.median():6.0f} {t.M.median():6.0f}"
              f"  {t.M.sum() / total:14.3f}")
    print("centroid dalam satuan asal (R hari, F faktur, M pound):")
    for k in range(4):
        print(f"{k + 1:6d}   R = {asal[k, 0]:5.0f}   F = {asal[k, 1]:5.1f}"
              f"   M = {asal[k, 2]:6.0f}")

    print()
    Z = fitur_log(tabel)
    for nama, model in (
            ("campuran Gaussian penuh", GaussianMixture(
                4, covariance_type="full", n_init=3,
                random_state=BENIH)),
            ("campuran Gaussian sferis", GaussianMixture(
                4, covariance_type="spherical", n_init=3,
                random_state=BENIH))):
        lab = model.fit(Z).predict(Z)
        print(f"ARI K-means lawan {nama:25s} "
              f"{adjusted_rand_score(label, lab):.3f}")
