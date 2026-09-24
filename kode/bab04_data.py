"""Bab 4: data yang dipakai sepanjang Bagian II.

delapan_titik()   : contoh hitungan tangan, titik A sampai H.
gumpalan(n, benih, sb): lima gumpalan normal di bidang, K = 5.
garis(benih)      : data satu dimensi untuk pemrograman dinamis.
"""
import numpy as np
from sklearn.datasets import make_blobs

BENIH = 20260924
NAMA = "ABCDEFGH"
PUSAT_GUMPALAN = np.array([[0, 0], [4, 0.5], [9, 1], [2, 5], [7, 6]],
                          dtype=float)


def delapan_titik():
    return np.array([[1, 0], [2, 1], [3, 1], [3, 2],
                     [5, 3], [6, 3], [8, 0], [8, 5]], dtype=float)


def gumpalan(n=500, benih=BENIH, sb=1.0):
    X, y = make_blobs(n_samples=n, centers=PUSAT_GUMPALAN,
                      cluster_std=sb, random_state=benih)
    return X, y


def garis(benih=BENIH):
    """Empat kelompok di garis bilangan, berjarak dan berukuran tidak
    sama, supaya K-means mudah terjebak."""
    rng = np.random.default_rng(benih)
    x = np.concatenate([rng.normal(0.0, 0.6, 30),
                        rng.normal(3.0, 0.4, 12),
                        rng.normal(4.6, 0.4, 12),
                        rng.normal(9.0, 1.0, 26)])
    return np.sort(x)
