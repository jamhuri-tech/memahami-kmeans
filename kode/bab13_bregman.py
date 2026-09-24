"""Bab 13: pusat terbaik untuk berbagai ukuran ketakmiripan.

Untuk 51 vektor positif berdimensi 3 (ganjil, supaya median tunggal), pusat m yang meminimumkan
jumlah ketakmiripan dicari secara numerik, lalu dibandingkan dengan
rata-rata, median per koordinat, dan rata-rata geometrik.
"""
import numpy as np
from scipy.optimize import minimize

from bab04_data import BENIH


def kl(x, m):          # divergensi Kullback-Leibler umum (Bregman)
    return (x * np.log(x / m) - x + m).sum(axis=-1)


def itakura_saito(x, m):                       # juga Bregman
    return (x / m - np.log(x / m) - 1).sum(axis=-1)


def kuadrat(x, m):
    return ((x - m) ** 2).sum(axis=-1)


def l1(x, m):                                  # bukan Bregman
    return np.abs(x - m).sum(axis=-1)


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X = rng.gamma(2.0, 1.5, size=(51, 3))
    acuan = {"rata-rata": X.mean(axis=0),
             "median": np.median(X, axis=0),
             "rata geometrik": np.exp(np.log(X).mean(axis=0)),
             "rata harmonik": 1 / (1 / X).mean(axis=0)}
    print("ketakmiripan       arah      jarak terdekat ke")
    for nama, f in (("kuadrat", kuadrat), ("KL", kl),
                    ("Itakura-Saito", itakura_saito), ("L1", l1)):
        for arah_nama, g in (("D(x, m)", lambda m: f(X, m).sum()),
                             ("D(m, x)", lambda m: f(m, X).sum())):
            if nama in ("kuadrat", "L1") and arah_nama == "D(m, x)":
                continue                       # setangkup
            m = minimize(g, X.mean(axis=0), method="Nelder-Mead",
                         options={"xatol": 1e-10, "fatol": 1e-12,
                                  "maxiter": 20000}).x
            dekat = min(acuan, key=lambda a: np.abs(m - acuan[a]).max())
            print(f"{nama:15s} {arah_nama}   {dekat:15s}"
                  f" {np.abs(m - acuan[dekat]).max():.1e}")
