"""Bab 10: memakai ukuran eksternal.

(1) digits: K = 5..30, bagaimana setiap ukuran berubah terhadap K.
(2) Dua partisi yang sama-sama benar: KMeansKita lawan scikit-learn
    pada benih digits yang berbeda di Bab 7, dan titik-titik tetap
    data gumpalan dari Bab 5.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_mutual_info_score

from bab04_data import BENIH, gumpalan
from bab05_lloyd import lloyd
from bab07_kmeans import KMeansKita
from bab10_ukuran import akurasi_hungaria, ari, nmi, purity

if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    print("(1) digits      purity  hungaria   ARI    NMI    AMI")
    for K in (5, 8, 10, 12, 15, 20, 30):
        lab = KMeans(K, n_init=10, random_state=BENIH).fit(X).labels_
        print(f"    K = {K:2d}     {purity(y, lab):6.3f}"
              f" {akurasi_hungaria(y, lab):9.3f} {ari(y, lab):6.3f}"
              f" {nmi(y, lab):6.3f}"
              f" {adjusted_mutual_info_score(y, lab):6.3f}")
    a = KMeansKita(10, random_state=9).fit(X)
    b = KMeans(10, init="random", random_state=9,
               algorithm="lloyd").fit(X)
    print(f"(2) digits benih 9, KMeansKita lawan sklearn:")
    print(f"      ARI {ari(a.labels_, b.labels_):.4f}, beda label "
          f"{(a.labels_ != b.labels_).sum()}")
    Xg, yg = gumpalan()
    rng = np.random.default_rng(BENIH)
    hasil = {}
    for _ in range(300):
        C, lab, _ = lloyd(Xg, Xg[rng.choice(len(Xg), 5, replace=False)])
        J = round(((Xg - C[lab]) ** 2).sum(), 2)
        hasil.setdefault(J, lab)
    Js = sorted(hasil)
    acuan = hasil[Js[0]]
    print(f"    gumpalan, titik tetap terbaik (J {Js[0]:.2f}) lawan:")
    for J in (Js[1], Js[2], Js[-1]):
        print(f"      J {J:8.2f}: ARI {ari(acuan, hasil[J]):.4f},"
              f" ARI thd label asli {ari(yg, hasil[J]):.4f}")
    print(f"      J {Js[0]:8.2f}: ARI thd label asli "
          f"{ari(yg, acuan):.4f}")
