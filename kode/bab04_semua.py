"""Bab 4: menelusuri semua partisi delapan titik.

partisi(n, K) membangkitkan setiap partisi n titik menjadi tepat K
klaster tak kosong, masing-masing satu kali (barisan label bertumbuh
terbatas). titik_tetap memeriksa apakah setiap titik paling dekat ke
centroid klasternya sendiri (Listing 4.2).
"""
from math import comb, factorial

import numpy as np

from bab04_data import NAMA, delapan_titik
from bab04_inersia import centroid, inersia


def partisi(n, K):
    label = [0] * n

    def isi(i, terpakai):
        if n - i < K - terpakai:           # sisa titik tidak cukup
            return
        if i == n:
            if terpakai == K:
                yield np.array(label)
            return
        for k in range(min(terpakai + 1, K)):
            label[i] = k
            yield from isi(i + 1, max(terpakai, k + 1))

    yield from isi(0, 0)


def titik_tetap(X, label, K):
    mu = centroid(X, label, K)
    D = ((X[:, None, :] - mu[None, :, :]) ** 2).sum(axis=2)
    sendiri = D[np.arange(len(X)), label]
    D[np.arange(len(X)), label] = np.inf
    return bool(np.all(sendiri < D.min(axis=1)))


def stirling2(n, K):
    return sum((-1) ** j * comb(K, j) * (K - j) ** n
               for j in range(K + 1)) // factorial(K)


def teks(label, K):
    return " | ".join("".join(NAMA[i] for i in np.flatnonzero(label == k))
                      for k in range(K))


if __name__ == "__main__":
    X = delapan_titik()
    for K in (2, 3):
        semua = [(inersia(X, p, K), p) for p in partisi(len(X), K)]
        semua.sort(key=lambda t: t[0])
        J = np.array([j for j, _ in semua])
        print(f"K = {K}: {len(semua)} partisi (S(8,{K}) = "
              f"{stirling2(8, K)}), J dari {J[0]:.2f} sampai "
              f"{J[-1]:.2f}")
        tetap = [(r, j, p) for r, (j, p) in enumerate(semua)
                 if titik_tetap(X, p, K)]
        print(f"  {len(tetap)} titik tetap:")
        for r, j, p in tetap:
            print(f"    peringkat {r + 1:3d}  J = {j:6.2f}  {teks(p, K)}")
    terbaik = [min(inersia(X, p, K) for p in partisi(len(X), K))
               for K in range(1, 9)]
    print("J terbaik K=1..8:",
          " ".join(f"{j:.2f}" for j in terbaik))
    print("banyaknya partisi S(n, K):")
    for n in (10, 20, 50, 100):
        baris = "  ".join(f"{float(stirling2(n, K)):.1e}"
                          for K in (2, 3, 5))
        print(f"  n = {n:3d}:  K=2,3,5 -> {baris}")
