"""Bab 3: argmin, rata-rata per klaster, dan blok dengan reshape.

rata_per_klaster dicetak di naskah (Listing 3.2): np.bincount menghitung
anggota setiap klaster, np.add.at menjumlahkan baris-baris X ke baris
klasternya. Hasilnya dibandingkan dengan perulangan biasa. Terakhir,
larik 4 x 6 dipotong menjadi blok 2 x 3 dengan reshape dan transpose.
"""
import numpy as np

from bab04_data import BENIH


def rata_per_klaster(X, label, K):
    nk = np.bincount(label, minlength=K)  # banyak anggota
    jumlah = np.zeros((K, X.shape[1]))
    np.add.at(jumlah, label, X)           # jumlah per klaster
    return jumlah / nk[:, None], nk


if __name__ == "__main__":
    D = np.array([[4.0, 1.0, 1.0],
                  [2.0, 3.0, 0.5]])
    print("D =", D.tolist())
    print("D.argmin(axis=1) =", D.argmin(axis=1).tolist())

    rng = np.random.default_rng(BENIH)
    X = rng.normal(size=(1000, 3))
    label = rng.integers(0, 4, size=1000)
    C, nk = rata_per_klaster(X, label, 4)
    C_loop = np.array([X[label == k].mean(axis=0) for k in range(4)])
    print(f"nk = {nk.tolist()}")
    print(f"maks |beda dengan perulangan| = {np.abs(C - C_loop).max():.1e}")
    salah = np.zeros((4, 3))
    salah[label] += X                             # indeks ganda!
    print(f"jumlah dengan +=: {salah[0, 0]:.3f}, "
          f"dengan add.at: {C[0, 0] * nk[0]:.3f}")
    with np.errstate(invalid="ignore"):
        C5, nk5 = rata_per_klaster(X, label, 5)
    print(f"K = 5, klaster ke-5 kosong: nk = {nk5[4]}, centroid = {C5[4]}")

    print()
    g = np.arange(24).reshape(4, 6)
    print("g =")
    print(g)
    B = g.reshape(2, 2, 2, 3).transpose(0, 2, 1, 3).reshape(-1, 6)
    print("blok 2 x 3, satu baris per blok:")
    print(B)
