"""Bab 5: algoritma Lloyd pada delapan titik, dengan tangan.

Centroid awal: titik A dan B. Setiap iterasi mencetak kuadrat jarak,
label, dan inersia sesudah penugasan serta sesudah pembaruan.
"""
import numpy as np
from sklearn.cluster import KMeans

from bab04_data import NAMA, delapan_titik
from bab05_lloyd import jarak2, lloyd


def baris(nama, v):
    return nama + "".join(f"{a:6.2f}" for a in v)


if __name__ == "__main__":
    X = delapan_titik()
    C = X[[0, 1]].copy()
    C_akhir, label_akhir, riwayat = lloyd(X, C)
    for t, (J_tugas, J_baru, C_baru, label) in enumerate(riwayat):
        print(f"iterasi {t + 1}: mu1 = ({C[0, 0]:.2f}, {C[0, 1]:.2f})"
              f"  mu2 = ({C[1, 0]:.2f}, {C[1, 1]:.2f})")
        D = jarak2(X, C)
        print("    " + "".join(f"{h:>6s}" for h in NAMA))
        print(baris("d1  ", D[:, 0]))
        print(baris("d2  ", D[:, 1]))
        print("lbl " + "".join(f"{k + 1:6d}" for k in label))
        print(f"J sesudah penugasan {J_tugas:.2f}, "
              f"sesudah pembaruan {J_baru:.2f}")
        C = C_baru
    print(f"iterasi {len(riwayat) + 1}: label tidak berubah, berhenti")
    km = KMeans(2, init=X[[0, 1]], n_init=1).fit(X)
    print("scikit-learn: label", (km.labels_ + 1).tolist())
    print(f"  centroid {km.cluster_centers_.round(4).tolist()}")
    print(f"  inertia_ {km.inertia_:.4f}, n_iter_ {km.n_iter_}")
