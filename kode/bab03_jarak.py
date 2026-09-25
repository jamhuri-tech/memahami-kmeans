"""Bab 3: tiga cara menghitung matriks kuadrat jarak n x K.

jarak2_loop, jarak2_siar (broadcasting), dan jarak2_jabar (jabaran
||x||^2 - 2 x.m + ||m||^2) dicetak di naskah (Listing 3.1). Hasil
ketiganya dibandingkan, lalu jabaran diuji pada data yang jauh dari
titik asal.
"""
import numpy as np

from bab04_data import BENIH


def jarak2_loop(X, C):
    D = np.empty((len(X), len(C)))
    for i in range(len(X)):
        for k in range(len(C)):
            D[i, k] = ((X[i] - C[k]) ** 2).sum()
    return D


def jarak2_siar(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def jarak2_jabar(X, C):
    xx = (X ** 2).sum(axis=1)[:, None]      # n x 1
    cc = (C ** 2).sum(axis=1)[None, :]      # 1 x K
    return xx - 2 * X @ C.T + cc            # n x K


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    X, C = rng.normal(size=(1000, 20)), rng.normal(size=(50, 20))
    D0 = jarak2_loop(X, C)
    print(f"X {X.shape}, C {C.shape}, D {D0.shape}")
    for nama, f in (("siaran", jarak2_siar), ("jabaran", jarak2_jabar)):
        print(f"  {nama:8s} maks |beda dengan loop| = "
              f"{np.abs(f(X, C) - D0).max():.1e}")
    n, K, d = 100_000, 50, 20
    print(f"larik antara siaran untuk n={n}, K={K}, d={d}: "
          f"{n * K * d * 8 / 1e6:.0f} MB")

    print()
    print("data digeser sejauh s dari titik asal:")
    print("        s   maks galat relatif jabaran   D terkecil")
    Xk = rng.normal(size=(1000, 2))
    Ck = Xk[:5] + 1e-3                   # centroid sangat dekat titik
    for s in (0.0, 1e2, 1e4, 1e6):
        benar = jarak2_siar(Xk + s, Ck + s)
        jabar = jarak2_jabar(Xk + s, Ck + s)
        rel = (np.abs(jabar - benar) / benar).max()
        print(f"  {s:7.0e}  {rel:24.1e}  {jabar.min():11.1e}")
