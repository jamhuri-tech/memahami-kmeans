"""Bab 5: inersia tidak pernah naik, diperiksa pada 200 awal acak.

Setiap awal memilih K titik data berbeda secara acak sebagai centroid.
Barisan J dicatat per setengah langkah: sesudah penugasan, sesudah
pembaruan, sesudah penugasan berikutnya, dan seterusnya.
"""
import numpy as np

from bab04_data import BENIH, gumpalan
from bab05_lloyd import lloyd


def barisan_J(riwayat):
    J = []
    for J_tugas, J_baru, _, _ in riwayat:
        J += [J_tugas, J_baru]
    return np.array(J)


if __name__ == "__main__":
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    naik, iterasi = [], []
    for _ in range(200):
        C0 = X[rng.choice(len(X), 5, replace=False)]
        _, _, riwayat = lloyd(X, C0)
        J = barisan_J(riwayat)
        naik.append(np.diff(J).max())
        iterasi.append(len(riwayat) + 1)
    naik = np.array(naik)
    print(f"200 awal acak, K = 5, n = {len(X)}")
    print(f"setengah langkah yang menaikkan J: {(naik > 0).sum()}")
    print(f"penurunan J terkecil: {-naik.max():.1e}")
    print(f"iterasi: min {min(iterasi)}, "
          f"median {np.median(iterasi):.0f}, maks {max(iterasi)}")
