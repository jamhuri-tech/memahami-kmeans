"""Bab 5: ke titik tetap mana algoritma Lloyd berhenti?

(1) Delapan titik, K = 2: semua 28 pasangan titik data sebagai awal.
(2) Data gumpalan, K = 5: 1000 awal acak dari titik data.
"""
from itertools import combinations

import numpy as np

from bab04_data import BENIH, NAMA, delapan_titik, gumpalan
from bab05_lloyd import lloyd


def inersia_akhir(X, C0):
    C, label, _ = lloyd(X, C0)
    return ((X - C[label]) ** 2).sum()


if __name__ == "__main__":
    X = delapan_titik()
    hasil = {}
    for i, j in combinations(range(8), 2):
        J = round(inersia_akhir(X, X[[i, j]]), 2)
        hasil.setdefault(J, []).append(NAMA[i] + NAMA[j])
    print("delapan titik, 28 pasangan awal:")
    for J in sorted(hasil):
        print(f"  J = {J:6.2f}: {len(hasil[J]):2d} awal  "
              f"{' '.join(hasil[J][:6])}"
              f"{' ...' if len(hasil[J]) > 6 else ''}")

    Xg, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    J = np.array([inersia_akhir(Xg, Xg[rng.choice(len(Xg), 5,
                                                   replace=False)])
                  for _ in range(1000)])
    terbaik = J.min()
    beda = np.unique(J.round(6))
    print(f"gumpalan, 1000 awal acak: {len(beda)} nilai J akhir")
    print(f"  J terbaik {terbaik:.2f}, dicapai "
          f"{np.isclose(J, terbaik).sum()} kali")
    print(f"  median {np.median(J):.2f}, terburuk {J.max():.2f} "
          f"({J.max() / terbaik:.2f} x terbaik)")
    print(f"  awal dengan J > 1.1 x terbaik: {(J > 1.1 * terbaik).sum()}")
