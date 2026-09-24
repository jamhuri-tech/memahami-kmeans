"""Bab 10: ukuran eksternal kita lawan sklearn.metrics."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score, rand_score)

from bab04_data import BENIH, gumpalan
from bab10_ukuran import ari, nmi, rand

if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    Xd, yd = load_digits(return_X_y=True)
    Xg, yg = gumpalan()
    pasangan = [
        ("digits, K=10", yd, KMeans(10, n_init=10, random_state=BENIH)
         .fit(Xd).labels_),
        ("digits, K=4", yd, KMeans(4, n_init=10, random_state=BENIH)
         .fit(Xd).labels_),
        ("gumpalan, K=5", yg, KMeans(5, n_init=10, random_state=BENIH)
         .fit(Xg).labels_),
        ("acak, K=10", yd, rng.integers(0, 10, len(yd)))]
    print("pasangan          |RI|      |ARI|     |NMI|     ARI")
    for nama, a, b in pasangan:
        e = [abs(rand(a, b) - rand_score(a, b)),
             abs(ari(a, b) - adjusted_rand_score(a, b)),
             abs(nmi(a, b) - normalized_mutual_info_score(a, b))]
        teks = "  ".join("< 1e-12 " if v < 1e-12 else f"{v:8.1e}"
                         for v in e)
        print(f"{nama:15s} {teks}  {ari(a, b):6.3f}")
