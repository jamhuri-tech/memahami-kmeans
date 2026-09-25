"""Bab 3: KMeans scikit-learn sekilas.

Atribut yang dipakai sepanjang buku: labels_, cluster_centers_,
inertia_, n_iter_, lalu predict, transform, dan score. Data: gumpalan
dari bab04_data (500 titik, lima gumpalan).
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH, gumpalan

if __name__ == "__main__":
    X, _ = gumpalan()
    km = KMeans(n_clusters=5, n_init=10, random_state=BENIH).fit(X)
    print("labels_[:10]      ", km.labels_[:10].tolist())
    print("cluster_centers_  ", km.cluster_centers_.shape)
    print(np.round(km.cluster_centers_, 3))
    print(f"inertia_           {km.inertia_:.4f}")
    print(f"n_iter_            {km.n_iter_}")
    J = ((X - km.cluster_centers_[km.labels_]) ** 2).sum()
    print(f"inersia dihitung   {J:.4f}")
    baru = np.array([[0.0, 0.0], [8.0, 5.0]])
    print("predict(baru)     ", km.predict(baru).tolist())
    print("transform(baru)   ", km.transform(baru).shape,
          "jarak ke setiap centroid")
    print(f"score(X)           {km.score(X):.4f}")
