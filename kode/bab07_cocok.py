"""Bab 7: KMeansKita lawan KMeans scikit-learn, angka demi angka.

Untuk setiap data dan setiap random_state 0..9, kedua kelas dilatih
dengan pengaturan yang sama (init="random", n_init bawaan = 10).
"""
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, load_iris, load_wine

from bab04_data import BENIH, delapan_titik, gumpalan
from bab07_kmeans import KMeansKita


def data_uji():
    rng = np.random.default_rng(BENIH)
    Xg, _ = gumpalan()
    Xg2, _ = gumpalan(5000, BENIH, 2.0)
    Xb = rng.normal(size=(2000, 20)) + \
        rng.normal(0, 3, size=(8, 20))[rng.integers(0, 8, 2000)]
    return [("gumpalan", Xg, 5, None),
            ("gumpalan sb=2", Xg2, 5, None),
            ("iris", load_iris().data, 3, None),
            ("wine", load_wine().data, 3, None),
            ("digits", load_digits().data, 10, None),
            ("20 dimensi", Xb, 8, None),
            ("gumpalan+bobot", Xg, 5, rng.uniform(0.1, 3, len(Xg)))]


def angka(v, batas=1e-12):
    """Selisih di bawah batas dicetak sebagai batas saja, karena
    penjumlahan paralel scikit-learn tidak selalu berurutan sama."""
    return f"{'< ' + format(batas, '.0e'):>9s}" if v < batas \
        else f"{v:9.1e}"


def bandingkan(X, K, w, s, **kw):
    a = KMeansKita(K, random_state=s, **kw).fit(X, sample_weight=w)
    b = KMeans(K, init=kw.get("init", "random"), random_state=s,
               algorithm="lloyd").fit(X, sample_weight=w)
    return (np.array_equal(a.labels_, b.labels_),
            np.abs(a.cluster_centers_ - b.cluster_centers_).max(),
            abs(a.inertia_ - b.inertia_) / b.inertia_,
            a.n_iter_ == b.n_iter_)


if __name__ == "__main__":
    print("data            K  label  n_iter  |dC| maks  dJ relatif")
    for nama, X, K, w in data_uji():
        hasil = [bandingkan(X, K, w, s) for s in range(10)]
        lab = sum(h[0] for h in hasil)
        it = sum(h[3] for h in hasil)
        dC = max(h[1] for h in hasil)
        dJ = max(h[2] for h in hasil)
        print(f"{nama:14s} {K:2d}  {lab:3d}/10 {it:4d}/10"
              f"  {angka(dC)}  {angka(dJ)}")
    X = delapan_titik()
    C0 = np.array([[1.0, 0.0], [8.0, 5.0], [20.0, 20.0]])
    a = KMeansKita(3, init=C0).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = KMeans(3, init=C0, n_init=1).fit(X)
    print(f"klaster kosong: label sama "
          f"{np.array_equal(a.labels_, b.labels_)}, "
          f"J {a.inertia_:.4f} lawan {b.inertia_:.4f}")
