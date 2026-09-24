"""Bab 9: empat data untuk memilih K.

gumpalan    : lima gumpalan terpisah (K sebenarnya 5)
tumpang     : lima gumpalan bersimpangan baku 2 (K sebenarnya 5)
seragam     : 500 titik seragam di persegi, tanpa klaster (K = 1)
digits      : 1797 citra angka 8 x 8, sepuluh kelas
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

from bab04_data import BENIH, gumpalan


def semua_data():
    rng = np.random.default_rng(BENIH)
    return [("gumpalan", gumpalan()[0], 5),
            ("tumpang", gumpalan(1000, BENIH, 2.0)[0], 5),
            ("seragam", rng.uniform(0, 10, size=(500, 2)), 1),
            ("digits", load_digits().data, 10)]


def kmeans(X, K, benih=BENIH):
    return KMeans(K, n_init=10, random_state=benih).fit(X)
