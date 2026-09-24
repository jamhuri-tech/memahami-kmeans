"""Bab 10: K-means pada digits dibandingkan dengan label angka."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

from bab04_data import BENIH
from bab10_ukuran import akurasi_hungaria, ari, nmi, purity, tabel

if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    lab = KMeans(10, n_init=10, random_state=BENIH).fit(X).labels_
    T = tabel(y, lab)
    baris, kolom = linear_sum_assignment(-T)
    T = T[:, kolom]                       # urutkan kolom menurut padanan
    print("angka  klaster (diurutkan menurut padanan Hungaria)")
    for d in range(10):
        print(f"  {d}   " + "".join(f"{v:5d}" for v in T[d]))
    print(f"akurasi tanpa pencocokan: {(lab == y).mean():.3f}")
    print(f"akurasi sesudah pencocokan Hungaria: "
          f"{akurasi_hungaria(y, lab):.3f}")
    print(f"purity {purity(y, lab):.3f}, ARI {ari(y, lab):.3f}, "
          f"NMI {nmi(y, lab):.3f}")
