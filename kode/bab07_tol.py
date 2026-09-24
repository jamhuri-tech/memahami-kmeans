"""Bab 7: toleransi relatif tol dan penugasan akhir.

Data gumpalan sb=2, 5000 titik, satu awal acak yang sama untuk
setiap tol. Kolom "beda" menghitung label yang berbeda dari hasil
tol = 0 (konvergensi ketat).
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH, gumpalan
from bab07_kmeans import KMeansKita

if __name__ == "__main__":
    X, _ = gumpalan(5000, BENIH, 2.0)
    v = np.var(X, axis=0).mean()
    print(f"rata-rata variansi per peubah: {v:.4f}")
    acuan = KMeansKita(5, n_init=1, tol=0, random_state=3).fit(X)
    print("    tol   tol mutlak  n_iter kita  sklearn        J  beda")
    for tol in (0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        a = KMeansKita(5, n_init=1, tol=tol, random_state=3).fit(X)
        b = KMeans(5, init="random", n_init=1, tol=tol,
                   random_state=3).fit(X)
        print(f"  {tol:5.0e}  {tol * v:11.2e}  {a.n_iter_:11d}"
              f"  {b.n_iter_:7d}  {a.inertia_:8.1f}"
              f"  {(a.labels_ != acuan.labels_).sum():4d}")
    a = KMeansKita(5, n_init=1, random_state=3).fit(X * 1000)
    print(f"data dikali 1000, tol 1e-4: n_iter {a.n_iter_}")
