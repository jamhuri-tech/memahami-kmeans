"""Bab 14: kernel K-means pada bulan, cincin, dan gumpalan.

Kernel RBF dengan beberapa gamma; 20 awal berupa label acak, disimpan
hasil berobjektif terkecil. Kolom 'awal baik' menghitung awal yang
mencapai ARI di atas 0.9.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

from bab11_data import enam_data
from bab14_kernel import rbf, terbaik

if __name__ == "__main__":
    data = {n: (X, y) for n, X, y in enam_data()}
    print("data      gamma   ARI terbaik-J   awal baik")
    for nama in ("gumpalan", "bulan", "cincin"):
        X, y = data[nama]
        K = y.max() + 1
        for g in (0.1, 1, 5, 20, 50):
            (lab, J), semua = terbaik(rbf(X, g), K)
            baik = sum(ari(y, l) > 0.9 for l, _ in semua)
            print(f"{nama:9s} {g:5g} {ari(y, lab):14.3f} {baik:8d}/20")
