"""Bab 10: ukuran eksternal dari tabel kontingensi.

tabel, akurasi_hungaria, purity (Listing 10.1) dan rand, ari,
nmi (Listing 10.2), dicocokkan dengan sklearn.metrics.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import comb


def tabel(a, b):
    """Tabel kontingensi: baris = label a, kolom = label b."""
    ia, a = np.unique(a, return_inverse=True)
    ib, b = np.unique(b, return_inverse=True)
    T = np.zeros((len(ia), len(ib)), dtype=int)
    np.add.at(T, (a, b), 1)
    return T


def akurasi_hungaria(benar, klaster):
    T = tabel(benar, klaster)
    baris, kolom = linear_sum_assignment(-T)   # maksimumkan
    return T[baris, kolom].sum() / T.sum()


def purity(benar, klaster):
    T = tabel(benar, klaster)
    return T.max(axis=0).sum() / T.sum()  # mayoritas


def rand(a, b):
    T = tabel(a, b)
    n = T.sum()
    sama_keduanya = comb(T, 2).sum()
    sama_a = comb(T.sum(axis=1), 2).sum()
    sama_b = comb(T.sum(axis=0), 2).sum()
    total = comb(n, 2)
    beda_keduanya = total - sama_a - sama_b + sama_keduanya
    return (sama_keduanya + beda_keduanya) / total


def ari(a, b):
    T = tabel(a, b)
    indeks = comb(T, 2).sum()
    sa = comb(T.sum(axis=1), 2).sum()
    sb = comb(T.sum(axis=0), 2).sum()
    harap = sa * sb / comb(T.sum(), 2)     # harapan kebetulan
    maks = (sa + sb) / 2
    return (indeks - harap) / (maks - harap)


def nmi(a, b):
    P = tabel(a, b) / len(a)
    pa, pb = P.sum(axis=1), P.sum(axis=0)
    ada = P > 0
    Q = np.outer(pa, pb)                   # jika saling bebas
    I = (P[ada] * np.log(P[ada] / Q[ada])).sum()
    Ha = -(pa * np.log(pa)).sum()
    Hb = -(pb * np.log(pb)).sum()
    return I / ((Ha + Hb) / 2)
