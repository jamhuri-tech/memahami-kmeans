"""Bab 6: lanskap F(mu1, mu2) pada data satu dimensi, K = 2.

(1) Titik tetap Lloyd dari kisi centroid awal.
(2) Setiap titik tetap adalah minimum lokal F: gangguan kecil acak
    tidak pernah menurunkan F.
(3) Pengganti MM: G(mu | mu_t) >= F(mu), sama di mu_t.
"""
import numpy as np

from bab05_lloyd import jarak2, lloyd
from bab06_fungsi import F, tiga_kelompok


def pengganti(X, C, C_t):
    """G(C | C_t): setiap titik tetap memakai label dari C_t."""
    label = jarak2(X, C_t).argmin(axis=1)
    return ((X - C[label]) ** 2).sum()


if __name__ == "__main__":
    X = tiga_kelompok()
    kisi = np.linspace(-1, 10, 23)
    tetap = {}
    for a in kisi:
        for b in kisi:
            if a >= b:
                continue
            C, label, _ = lloyd(X, np.array([[a], [b]]))
            if len(set(label)) < 2:
                continue
            C = np.sort(C, axis=0)
            kunci = round(F(X, C), 4)
            tetap.setdefault(kunci, [C.ravel(), 0])
            tetap[kunci][1] += 1
    print(f"(1) {sum(v[1] for v in tetap.values())} awal, "
          f"{len(tetap)} titik tetap:")
    rng = np.random.default_rng(1)
    for J in sorted(tetap):
        C, m = tetap[J]
        C = C.reshape(-1, 1)
        J0 = F(X, C)
        naik = min(F(X, C + 1e-3 * rng.normal(size=C.shape)) - J0
                   for _ in range(2000))
        print(f"    ({C[0, 0]:.3f}, {C[1, 0]:.3f})  J {J:8.4f}"
              f"  {m:3d} awal  naik min {naik:.1e}")
    g = np.linspace(-2, 11, 131)
    selisih = []
    for C_t in (np.array([[0.0], [3.0]]), np.array([[2.0], [8.0]])):
        for a in g:
            for b in g:
                C = np.array([[a], [b]])
                selisih.append(pengganti(X, C, C_t) - F(X, C))
        print(f"(3) mu_t = {C_t.ravel().tolist()}: "
              f"G - F di mu_t = {pengganti(X, C_t, C_t) - F(X, C_t)}")
    print(f"    min G - F pada kisi 131 x 131: {min(selisih):.1f}")
