"""Bab 14: spectral clustering dari nol (Ng, Jordan, dan Weiss).

spektral dicetak di naskah (Listing 14.2): graf RBF, Laplacian
ternormalkan, K vektor eigen terkecil, baris dinormalkan, lalu K-means.
Dibandingkan dengan SpectralClustering scikit-learn lewat ARI.
"""
import warnings

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari

from bab04_data import BENIH
from bab11_data import enam_data
from bab14_kernel import rbf


def spektral(X, K, gamma, benih=BENIH):
    W = rbf(X, gamma)
    np.fill_diagonal(W, 0)             # tanpa sisi ke diri
    d = W.sum(axis=1)
    L = np.eye(len(X)) - W / np.sqrt(np.outer(d, d))
    nilai, vektor = np.linalg.eigh(L)  # urut naik
    U = vektor[:, :K]                  # K terkecil
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    km = KMeans(K, n_init=10, random_state=benih)
    lab = km.fit(U).labels_
    return lab, nilai[:K + 1]


if __name__ == "__main__":
    data = {n: (X, y) for n, X, y in enam_data()}
    print("data      gamma   kita  sklearn  kita-sklearn  celah")
    for nama in ("gumpalan", "bulan", "cincin"):
        X, y = data[nama]
        K = y.max() + 1
        for g in (1, 5, 20, 50):
            lab, nilai = spektral(X, K, g)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sk = SpectralClustering(K, affinity="rbf", gamma=g,
                                        random_state=BENIH).fit(X).labels_
            if w:
                print(f"  ({len(w)} peringatan lobpcg dari scikit-learn)")
            a, b, c = (round(v, 3) + 0.0 for v in
                       (ari(y, lab), ari(y, sk), ari(lab, sk)))
            print(f"{nama:9s} {g:5g} {a:6.3f} {b:8.3f}"
                  f" {c:13.3f} {nilai[K] - nilai[K - 1]:6.3f}")
