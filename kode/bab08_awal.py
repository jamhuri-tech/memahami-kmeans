"""Bab 8: lima cara memilih centroid awal, 1000 awal masing-masing.

forgy      : K titik data berbeda, acak seragam (init="random")
partisi    : label acak, centroid = rata-rata tiap kelompok
kotak      : titik acak seragam di kotak pembatas data
pp_asli    : k-means++ asli (sampling D^2, satu calon)
pp_serakah : k-means++ serakah scikit-learn (2 + ln K calon)
"""
import numpy as np

from bab04_data import BENIH, gumpalan
from bab06_fungsi import F
from bab07_kmeans import lloyd_satu
from bab08_plusplus import kmeanspp


def forgy(Xc, K, rng, w):
    return Xc[rng.choice(len(Xc), K, replace=False)]


def partisi(Xc, K, rng, w):
    label = rng.randint(K, size=len(Xc))
    return np.array([Xc[label == k].mean(axis=0) for k in range(K)])


def kotak(Xc, K, rng, w):
    return rng.uniform(Xc.min(axis=0), Xc.max(axis=0),
                       size=(K, Xc.shape[1]))


def pp_asli(Xc, K, rng, w):
    return kmeanspp(Xc, K, rng, w, calon=1)


def pp_serakah(Xc, K, rng, w):
    return kmeanspp(Xc, K, rng, w)


CARA = [("forgy", forgy), ("partisi", partisi), ("kotak", kotak),
        ("pp_asli", pp_asli), ("pp_serakah", pp_serakah)]


def kisi(n_per=50, benih=BENIH):
    """25 gumpalan pada kisi 5 x 5 berjarak 4, simpangan baku 0,7."""
    rng = np.random.default_rng(benih)
    pusat = np.array([(4.0 * i, 4.0 * j) for i in range(5)
                      for j in range(5)])
    return np.vstack([c + 0.7 * rng.normal(size=(n_per, 2))
                      for c in pusat])


def jalankan(X, K, cara, ulang=1000):
    Xc = X - X.mean(axis=0)
    w = np.ones(len(X))
    rng = np.random.RandomState(BENIH)
    awal, akhir, it = [], [], []
    for _ in range(ulang):
        C0 = cara(Xc, K, rng, w)
        awal.append(F(Xc, C0))
        label, J, C, n_iter = lloyd_satu(Xc, w, C0, 300, 0.0)
        akhir.append(J)
        it.append(n_iter)
    return np.array(awal), np.array(akhir), np.array(it)


if __name__ == "__main__":
    for nama_data, X, K in (("gumpalan", gumpalan()[0], 5),
                            ("kisi 5x5", kisi(), 25)):
        hasil = {nama: jalankan(X, K, f) for nama, f in CARA}
        terbaik = min(h[1].min() for h in hasil.values())
        print(f"{nama_data}, K = {K}, J terbaik {terbaik:.2f}")
        print("  cara        J awal med  J akhir med  >1.1x  iter med")
        for nama, (a, j, it) in hasil.items():
            print(f"  {nama:10s} {np.median(a):11.1f} {np.median(j):12.2f}"
                  f" {(j > 1.1 * terbaik).sum():6d} {np.median(it):9.1f}")
