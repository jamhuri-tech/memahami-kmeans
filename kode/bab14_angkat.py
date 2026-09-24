"""Bab 14: membawa data cincin ke ruang fitur secara eksplisit.

phi(x) = (x1, x2, c * (x1^2 + x2^2)). Dengan c cukup besar, koordinat
ketiga menguasai jarak, dan kedua cincin terpisah menurut jari-jari.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

from bab04_data import BENIH
from bab11_data import enam_data

if __name__ == "__main__":
    _, X, y = enam_data()[5]
    print("cincin, K-means pada (x1, x2, c r^2):")
    for c in (0, 0.5, 1, 2, 5):
        F = np.column_stack([X, c * (X ** 2).sum(axis=1)])
        lab = KMeans(2, n_init=10, random_state=BENIH).fit(F).labels_
        print(f"  c = {c:3g}: ARI {round(ari(y, lab), 3) + 0.0:.3f}")
