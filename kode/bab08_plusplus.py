"""Bab 8: k-means++ dari nol, serakah seperti scikit-learn.

Fungsi kmeanspp dicetak di naskah (Listing 8.1). Tanda tangannya
cocok dengan argumen init KMeansKita: init(Xc, K, rng, w). Dengan
calon=1, fungsi ini menjadi k-means++ asli (sampling D^2 murni).
"""
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus

from bab04_data import BENIH, gumpalan
from bab07_kmeans import KMeansKita


def jarak2_ke(Xc, xx, idx):
    """Kuadrat jarak titik-titik Xc[idx] ke semua titik."""
    D = -2 * (Xc[idx] @ Xc.T)
    D += xx[idx][:, None]
    D += xx[None, :]
    return np.maximum(D, 0)


def kmeanspp(Xc, K, rng, w, calon=None):
    n = len(Xc)
    if calon is None:
        calon = 2 + int(np.log(K))      # serakah
    xx = np.einsum("ij,ij->i", Xc, Xc)
    idx = [rng.choice(n, p=w / w.sum())]
    d2 = jarak2_ke(Xc, xx, idx)[0]      # ke centroid terdekat
    potensi = d2 @ w
    for _ in range(1, K):
        u = rng.uniform(size=calon) * potensi
        pilih = np.searchsorted(np.cumsum(w * d2), u)
        pilih = np.minimum(pilih, n - 1)
        D = np.minimum(d2, jarak2_ke(Xc, xx, pilih))
        pot = D @ w                     # potensi tiap calon
        b = pot.argmin()
        idx.append(pilih[b])
        d2, potensi = D[b], pot[b]
    return Xc[idx]


def kmeanspp_asli(Xc, K, rng, w):
    return kmeanspp(Xc, K, rng, w, calon=1)


if __name__ == "__main__":
    X, _ = gumpalan()
    w = np.ones(len(X))
    sama = 0
    for s in range(100):
        C, _ = kmeans_plusplus(X, 5, random_state=s)
        C_kita = kmeanspp(X, 5, np.random.RandomState(s), w)
        sama += np.array_equal(C, C_kita)
    print(f"kmeans_plusplus lawan kmeanspp, 100 benih: {sama} sama")
    hasil = []
    for s in range(20):
        a = KMeansKita(5, init=kmeanspp, random_state=s).fit(X)
        b = KMeans(5, random_state=s).fit(X)
        hasil.append(np.array_equal(a.labels_, b.labels_)
                     and a.n_iter_ == b.n_iter_)
    print(f"KMeansKita(init=kmeanspp) lawan KMeans, "
          f"20 benih: {sum(hasil)} sama")
