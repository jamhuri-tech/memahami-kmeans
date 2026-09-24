"""Bab 12: empat jenis kovarians lawan K-means pada enam data Bab 11.

Setiap GaussianMixture memakai n_init=5; K sama dengan K sebenarnya.
Kolom terakhir: jenis kovarians yang dipilih BIC.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH
from bab11_data import enam_data

JENIS = ("spherical", "diag", "tied", "full")

if __name__ == "__main__":
    print("data      K-means  sferis  diag  terikat  penuh  BIC")
    for nama, X, y in enam_data():
        K = y.max() + 1
        baris = [ari(y, KMeans(K, n_init=10, random_state=BENIH)
                     .fit(X).labels_)]
        bic = {}
        for j in JENIS:
            gm = GaussianMixture(K, covariance_type=j, n_init=5,
                                 random_state=BENIH).fit(X)
            baris.append(ari(y, gm.predict(X)))
            bic[j] = gm.bic(X)
        pilih = min(bic, key=bic.get)
        print(f"{nama:9s} " + " ".join(f"{round(v, 3) + 0.0:6.3f}"
                                      for v in baris) + f"  {pilih}")
