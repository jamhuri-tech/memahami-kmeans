"""Bab 6: gradien, Hessian, dan langkah Newton pada data gumpalan.

(1) Gradien rumus lawan beda hingga.
(2) Langkah Newton lawan langkah pembaruan Lloyd.
(3) Gradient descent dengan satu ukuran langkah untuk semua centroid.
"""
import numpy as np

from bab04_data import BENIH, gumpalan
from bab05_lloyd import jarak2, lloyd
from bab06_fungsi import F, gradien_hessian, langkah_newton

if __name__ == "__main__":
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    C0 = X[rng.choice(len(X), 5, replace=False)]

    g, h = gradien_hessian(X, C0)
    eps = 1e-6
    g_bh = np.zeros_like(C0)
    for k in range(5):
        for j in range(2):
            E = np.zeros_like(C0)
            E[k, j] = eps
            g_bh[k, j] = (F(X, C0 + E) - F(X, C0 - E)) / (2 * eps)
    print(f"(1) gradien: selisih relatif rumus vs beda hingga "
          f"{np.abs(g - g_bh).max() / np.abs(g).max():.1e}")
    print(f"    Hessian 2 n_k: {h.tolist()}")

    label = jarak2(X, C0).argmin(axis=1)
    C_lloyd = np.array([X[label == k].mean(axis=0) for k in range(5)])
    print(f"(2) |Newton - Lloyd| terbesar: "
          f"{np.abs(langkah_newton(X, C0) - C_lloyd).max():.1e}")

    C_akhir, _, riwayat = lloyd(X, C0)
    J_akhir = F(X, C_akhir)
    print(f"(3) Lloyd: {len(riwayat)} langkah ke J = {J_akhir:.4f}")
    nmaks = h.max() / 2
    print("    gradient descent, eta = c / (2 n_maks), "
          f"n_maks = {nmaks:.0f}")
    print("       c   langkah   J akhir")
    for c in (0.1, 0.5, 1.0, 1.5, 1.9, 2.1, 4.0):
        C = C0.copy()
        eta = c / (2 * nmaks)
        with np.errstate(all="ignore"):
            for t in range(1, 5001):
                g, _ = gradien_hessian(X, C)
                if not np.abs(g).max() >= 1e-8:   # juga jika nan
                    break
                C = C - eta * g
            J = F(X, C)
        if not np.isfinite(J):
            print(f"    {c:4.1f}  menyebar (centroid tak hingga)")
        else:
            print(f"    {c:4.1f} {t - 1:9d}   {J:10.4f}")
