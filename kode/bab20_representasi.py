"""Bab 20: K-means pada representasi yang berbeda.

Data digits (K = 10): ARI K-means pada piksel mentah, PCA 10 dimensi,
embedding spektral 10 dimensi, dan t-SNE 2 dimensi (tiga benih).
Lalu data seragam 10 dimensi tanpa klaster: silhouette K-means di
ruang asal dan di ruang t-SNE.
"""
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, silhouette_score

from bab04_data import BENIH


def label_km(Z, K):
    return KMeans(K, n_init=10, random_state=BENIH).fit(Z).labels_


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    X, y = load_digits(return_X_y=True)
    ruang = [("piksel mentah (64 dimensi)", X),
             ("PCA, 10 dimensi", PCA(10, random_state=BENIH)
              .fit_transform(X)),
             ("embedding spektral, 10 dimensi", SpectralEmbedding(
                 10, n_neighbors=10, random_state=BENIH)
              .fit_transform(X))]
    for s in range(3):
        ruang.append((f"t-SNE 2 dimensi, benih {s}",
                      TSNE(2, init="pca", random_state=s)
                      .fit_transform(X)))
    print("digits, K = 10                    ARI")
    for nama, Z in ruang:
        print(f"{nama:32s} {adjusted_rand_score(y, label_km(Z, 10)):.3f}")

    print()
    rng = np.random.default_rng(BENIH)
    U = rng.uniform(size=(1500, 10))
    T = TSNE(2, init="pca", random_state=0).fit_transform(U)
    print("seragam 10 dimensi, 1500 titik: silhouette K-means")
    print("   K   ruang asal   ruang t-SNE")
    for K in (5, 10):
        print(f"{K:4d} {silhouette_score(U, label_km(U, K)):12.3f}"
              f" {silhouette_score(T, label_km(T, K)):13.3f}")
