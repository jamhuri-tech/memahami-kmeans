"""Bab 13: empat metode pada data gumpalan dengan m pencilan.

Kolom terakhir: trimmed K-means dengan awal Forgy, bukan k-means++.

Pencilan diletakkan sejauh 40 dari pusat data, ke arah acak. Setiap
metode memakai 10 awal k-means++ yang sama dan menyimpan hasil
berobjektif terkecil. ARI dihitung pada titik gumpalan saja.
"""
import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import adjusted_rand_score as ari

from bab04_data import BENIH, gumpalan
from bab05_lloyd import lloyd
from bab13_kokoh import kmedians, kmedoids, trimmed_kmeans


def dengan_pencilan(m, jarak=40.0, benih=BENIH):
    X, y = gumpalan()
    rng = np.random.default_rng(benih)
    sudut = rng.uniform(0, 2 * np.pi, m)
    luar = np.column_stack([4.5 + jarak * np.cos(sudut),
                            3 + jarak * np.sin(sudut)])
    return np.vstack([X, luar]), y


def terbaik(fungsi, X, K, ulang=10, awal="pp"):
    hasil = None
    for s in range(ulang):
        if awal == "pp":
            C0, idx = kmeans_plusplus(X, K, random_state=s)
        else:                                  # Forgy
            idx = np.random.RandomState(s).choice(len(X), K,
                                                  replace=False)
            C0 = X[idx]
        h = fungsi(X, C0, idx)
        if hasil is None or h[-1] < hasil[-1]:
            hasil = h
    return hasil


def metode(X, K):
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))

    def km(X, C0, idx):
        C, lab, _ = lloyd(X, C0)
        return lab, ((X - C[lab]) ** 2).sum()

    def kmed(X, C0, idx):
        lab, C, J = kmedians(X, C0)
        return lab, J

    def kmd(X, C0, idx):
        lab, med, J = kmedoids(D, idx)
        return lab, J

    def trim(X, C0, idx):
        lab, C, J, buang = trimmed_kmeans(X, C0, 0.05)
        return lab, J

    return [("K-means", km, "pp"), ("K-medians", kmed, "pp"),
            ("K-medoids", kmd, "pp"), ("trimmed", trim, "pp"),
            ("trim+Forgy", trim, "forgy")]


if __name__ == "__main__":
    K = 5
    print("  m" + "".join(f"{n[0]:>11s}" for n in metode(
        dengan_pencilan(0)[0], K)))
    for m in (0, 5, 20):
        X, y = dengan_pencilan(m)
        baris = []
        for nama, f, awal in metode(X, K):
            lab = terbaik(f, X, K, awal=awal)[0]
            baris.append(ari(y, lab[:len(y)]))
        print(f"{m:3d}" + "".join(f"{v:11.3f}" for v in baris))
