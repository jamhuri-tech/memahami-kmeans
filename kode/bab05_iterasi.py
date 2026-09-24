"""Bab 5: berapa iterasi yang diperlukan algoritma Lloyd?

Data gumpalan dengan n berbeda dan dua simpangan baku. Setiap sel:
20 awal acak, tol = 0 (berhenti hanya jika label tidak berubah).
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH, gumpalan

if __name__ == "__main__":
    print("       n   sb=1: median  maks   sb=2: median  maks")
    for n in (100, 1000, 10000, 100000):
        sel = []
        for sb in (1.0, 2.0):
            X, _ = gumpalan(n, BENIH, sb)
            it = [KMeans(5, init="random", n_init=1, tol=0,
                         random_state=s).fit(X).n_iter_
                  for s in range(20)]
            sel += [np.median(it), max(it)]
        print(f"  {n:6d}   {sel[0]:10.1f} {sel[1]:5d}   "
              f"{sel[2]:10.1f} {sel[3]:5d}")
