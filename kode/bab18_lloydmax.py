"""Bab 18: kuantisasi skalar Lloyd-Max untuk sebaran normal baku.

lloyd_max dicetak di naskah (Listing 18.1). Algoritma Lloyd pada
sebaran kontinu: ambang = titik tengah antara dua level berurutan,
level = rata-rata bersyarat di antara dua ambang. Dibandingkan dengan
KMeans pada sampel, rumus Panter-Dite, dan kerapatan level phi^(1/3).
"""
import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

from bab04_data import BENIH


def momen(a, b):
    """Momen ke-0, 1, 2 sebaran normal baku di selang (a, b)."""
    fa, fb = norm.pdf(a), norm.pdf(b)
    m0 = norm.cdf(b) - norm.cdf(a)
    m1 = fa - fb
    with np.errstate(invalid="ignore"):     # inf * 0 di ujung
        m2 = m0 + np.nan_to_num(a * fa) - np.nan_to_num(b * fb)
    return m0, m1, m2


def lloyd_max(K, tol=1e-12, maks_iter=100000):
    c = norm.ppf((np.arange(K) + 0.5) / K)  # awal: kuantil
    for it in range(maks_iter):
        t = np.r_[-np.inf, (c[1:] + c[:-1]) / 2, np.inf]
        m0, m1, _ = momen(t[:-1], t[1:])
        c_baru = m1 / m0                # rata-rata bersyarat
        if np.abs(c_baru - c).max() < tol:
            break
        c = c_baru
    m0, m1, m2 = momen(t[:-1], t[1:])
    D = (m2 - 2 * c_baru * m1 + c_baru ** 2 * m0).sum()
    return c_baru, t, D, it + 1


if __name__ == "__main__":
    print("   K  iterasi   level positif               D")
    hasil = {}
    for K in (2, 4, 8, 16, 32):
        c, t, D, it = lloyd_max(K)
        hasil[K] = (c, D)
        pos = " ".join(f"{v:.4f}" for v in c[c > 0][:3])
        if (c > 0).sum() > 3:
            pos += " ..."
        print(f"{K:4d} {it:8d}   {pos:26s} {D:.6f}")

    print()
    rng = np.random.default_rng(BENIH)
    x = rng.standard_normal(200_000)[:, None]
    print("   K   D Lloyd-Max  D KMeans sampel  "
          "maks |beda level|")
    for K in (2, 4, 8, 16):
        c, D = hasil[K]
        km = KMeans(K, n_init=1, random_state=BENIH,
                    init=c[:, None]).fit(x)
        cs = np.sort(km.cluster_centers_.ravel())
        print(f"{K:4d}   {D:.6f}      {km.inertia_ / len(x):.6f}"
              f"           {np.abs(cs - c).max():.4f}")

    print()
    PD = np.sqrt(3) * np.pi / 2
    print("   K   K^2 D   (Panter-Dite: %.4f)" % PD)
    for K in (2, 4, 8, 16, 32):
        print(f"{K:4d}   {K * K * hasil[K][1]:.4f}")

    print()
    c = hasil[32][0]
    ramal = np.sqrt(3) * norm.ppf((np.arange(32) + 0.5) / 32)
    print("K = 32, level positif (ke-17 sampai ke-32):")
    for a in (16, 24):
        print("  Lloyd-Max:", " ".join(f"{v:.2f}" for v in c[a:a + 8]))
        print("  phi^(1/3):", " ".join(f"{v:.2f}"
                                       for v in ramal[a:a + 8]))
