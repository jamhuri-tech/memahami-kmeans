"""Bab 8: dua pemeriksaan tambahan.

(1) Data satu dimensi Bab 4: seberapa sering k-means++ sampai ke
    optimum yang dihitung pemrograman dinamis.
(2) Pencilan: lima titik jauh ditambahkan ke data gumpalan.
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import BENIH, garis, gumpalan
from bab04_satudimensi import kmeans_1d_tepat

if __name__ == "__main__":
    x = garis()
    J_opt, _ = kmeans_1d_tepat(x, 4)
    print(f"(1) garis, K = 4, optimum {J_opt:.4f}, 1000 awal:")
    for init in ("random", "k-means++"):
        J = np.array([KMeans(4, init=init, n_init=1, random_state=s)
                      .fit(x[:, None]).inertia_ for s in range(1000)])
        print(f"    {init:9s}: {np.isclose(J, J_opt).sum():4d} optimum,"
              f" {(J > 1.1 * J_opt).sum():4d} di atas 1.1 x")

    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    sudut = rng.uniform(0, 2 * np.pi, 5)
    jauh = np.column_stack([4.5 + 40 * np.cos(sudut),
                            3 + 40 * np.sin(sudut)])
    Xp = np.vstack([X, jauh])
    pencilan = np.arange(len(X), len(Xp))
    print("(2) gumpalan + 5 pencilan berjarak 40, K = 5, 1000 awal:")
    for init in ("random", "k-means++"):
        sendiri, J = 0, []
        for s in range(1000):
            km = KMeans(5, init=init, n_init=1, random_state=s).fit(Xp)
            lab = km.labels_
            ada = [k for k in range(5)
                   if set(np.flatnonzero(lab == k)) <= set(pencilan)]
            sendiri += len(ada) > 0
            J.append(km.inertia_)
        print(f"    {init:9s}: {sendiri:4d} dengan klaster pencilan,"
              f" J median {np.median(J):.1f}")
    km = KMeans(5, n_init=50, random_state=0).fit(Xp)
    print(f"    terbaik (n_init=50): J {km.inertia_:.1f}")
    print(f"    ukuran klaster {sorted(np.bincount(km.labels_).tolist())}")
