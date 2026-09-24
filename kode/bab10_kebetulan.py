"""Bab 10: ukuran eksternal untuk partisi acak terhadap label digits.

Untuk setiap K, 20 pelabelan acak seragam; dicetak rata-ratanya.
Partisi acak tidak mengandung informasi apa pun tentang angka.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_mutual_info_score

from bab04_data import BENIH
from bab10_ukuran import akurasi_hungaria, ari, nmi, purity, rand

if __name__ == "__main__":
    _, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(BENIH)
    print("    K    RI  purity  hungaria   NMI     ARI     AMI")
    for K in (2, 5, 10, 20, 50, 100, 300):
        h = np.array([[f(y, lab) for f in (rand, purity,
                                           akurasi_hungaria, nmi, ari,
                                           adjusted_mutual_info_score)]
                      for lab in (rng.integers(0, K, len(y))
                                  for _ in range(20))])
        m = np.round(h.mean(axis=0), 4) + 0.0  # tanpa -0
        print(f"  {K:3d} {m[0]:5.3f} {m[1]:7.3f} {m[2]:9.3f}"
              f" {m[3]:6.3f} {m[4]:7.4f} {m[5]:7.4f}")
