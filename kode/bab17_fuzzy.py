"""Bab 17: fuzzy c-means dari nol.

fcm dicetak di naskah (Listing 17.1). Keanggotaan u_ik di antara 0
dan 1, berjumlah satu untuk setiap titik; m > 1 mengatur kekaburan.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH, gumpalan


def fcm(X, C, m=2.0, tol=1e-9, maks_iter=1000):
    for it in range(maks_iter):
        d2 = ((X[:, None] - C[None]) ** 2).sum(axis=2)
        d2 = np.maximum(d2, 1e-300)    # titik tepat di pusat
        w = d2 ** (-1 / (m - 1))
        U = w / w.sum(axis=1, keepdims=True)  # keanggotaan
        Um = U ** m
        C_baru = Um.T @ X / Um.sum(axis=0)[:, None]
        if np.abs(C_baru - C).max() < tol:
            break
        C = C_baru
    return U, C_baru, it + 1


if __name__ == "__main__":
    X, y = gumpalan()
    km = KMeans(5, n_init=10, random_state=BENIH).fit(X)
    C0 = km.cluster_centers_
    print("   m   iterasi  |mu - mu_KMeans|  rata u terbesar   ARI")
    for m in (1.1, 1.5, 2, 3, 5):
        U, C, it = fcm(X, C0, m)
        print(f"{m:4g} {it:9d} {np.abs(C - C0).max():16.4f}"
              f" {U.max(axis=1).mean():16.3f} {ari(y, U.argmax(1)):6.3f}")
    U, C, _ = fcm(X, C0, 2)
    gm = GaussianMixture(5, covariance_type="spherical",
                         means_init=C0, random_state=BENIH).fit(X)
    for nama, p in (("tengah", (4.5, 3.0)), ("jauh", (60.0, 60.0))):
        w = ((np.array(p) - C) ** 2).sum(axis=1) ** -1.0
        g = gm.predict_proba([p])[0]
        print(f"titik {nama:6s} {str(p):12s} FCM u   = "
              + " ".join(f"{v:.2f}" for v in w / w.sum()))
        print(f"{'':25s} GMM gamma = "
              + " ".join(f"{v:.2f}" for v in g))
        print(f"{'':25s} GMM log p(x) = {gm.score_samples([p])[0]:.1f}")
