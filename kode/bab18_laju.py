"""Bab 18: distorsi lawan banyaknya centroid pada data seragam.

Untuk d = 1, 2, 3, 5, 10, 20: KMeans pada 20.000 titik seragam di
kubus satuan, K = 4 sampai 256, distorsi per dimensi pada data latih
dan pada 100.000 titik uji baru. Kemiringan log D terhadap log K
dibandingkan dengan -2/d. Lalu d = 2, K = 256, dengan 25.000 sampai
400.000 titik latih: tetapan D*K latih dan uji dibandingkan dengan
kisi persegi dan heksagon, serta banyaknya sisi sel Voronoi di dalam.
"""
import numpy as np
from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

from bab04_data import BENIH

DAFTAR_K = [4, 8, 16, 32, 64, 128, 256]


def kurva(d, rng, n=20_000, n_uji=100_000):
    X = rng.uniform(size=(n, d))
    Y = rng.uniform(size=(n_uji, d))
    latih, uji = [], []
    for K in DAFTAR_K:
        km = KMeans(K, n_init=3, random_state=BENIH).fit(X)
        latih.append(km.inertia_ / n / d)
        uji.append(-km.score(Y) / n_uji / d)
    return np.array(latih), np.array(uji)


def kemiringan(D):
    """Kemiringan log D terhadap log K untuk K = 16 sampai 256."""
    return np.polyfit(np.log(DAFTAR_K[2:]), np.log(D[2:]), 1)[0]


def sisi_dalam(C, tepi=0.02):
    """Banyaknya sisi sel Voronoi yang seluruhnya di dalam persegi."""
    v = Voronoi(C)
    sisi = []
    for r in v.point_region:
        reg = v.regions[r]
        if -1 in reg:
            continue
        P = v.vertices[reg]
        if (P > tepi).all() and (P < 1 - tepi).all():
            sisi.append(len(reg))
    return np.array(sisi)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    print("  d   miring latih  miring uji   -2/d")
    for d in (1, 2, 3, 5, 10, 20):
        latih, uji = kurva(d, rng)
        print(f"{d:3d}   {kemiringan(latih):11.3f} "
              f"{kemiringan(uji):11.3f} {-2 / d:7.3f}")

    print()
    Y = rng.uniform(size=(400_000, 2))
    print("d = 2, K = 256:")
    print("  n latih  D*K latih  D*K uji  sel dalam  bersisi 5/6/7")
    for n in (25_000, 100_000, 400_000):
        X = rng.uniform(size=(n, 2))
        km = KMeans(256, n_init=1, random_state=BENIH).fit(X)
        s = sisi_dalam(km.cluster_centers_)
        hit = "/".join(str((s == j).sum()) for j in (5, 6, 7))
        Dl = km.inertia_ / n * 256
        Du = -km.score(Y) / len(Y) * 256
        print(f"{n:9d}  {Dl:9.4f} {Du:8.4f}  {len(s):9d}  {hit:>13s}")
    print(f"kisi persegi 16 x 16: {1 / 6:.4f}   "
          f"heksagon: {5 / (18 * np.sqrt(3)):.4f}")
