"""Bab 16: KMeans lawan MiniBatchKMeans pada 500 000 titik.

Lima puluh gumpalan berdimensi 10, lima benih untuk setiap cara.
Inersia dihitung pada seluruh data dan dibandingkan dengan inersia
terkecil di antara semua jalan. Waktu diukur di gen_gambar.py
(bab16_waktu) dan hanya dilaporkan di prosa.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

from bab04_data import BENIH


def data_besar():
    X, _ = make_blobs(500000, n_features=10, centers=50,
                      cluster_std=2.0, center_box=(-20, 20),
                      random_state=BENIH)
    return X


if __name__ == "__main__":
    X = data_besar()
    hasil = {"KMeans": [(KMeans(50, n_init=1, random_state=s).fit(X)
                         .inertia_, np.nan) for s in range(5)]}
    for b in (256, 1024, 4096):
        hasil[f"mini-batch {b}"] = []
        for s in range(5):
            mb = MiniBatchKMeans(50, batch_size=b, n_init=1,
                                 random_state=s).fit(X)
            hasil[f"mini-batch {b}"].append(
                (-mb.score(X), mb.n_steps_ * b / len(X)))
    J_min = min(j for r in hasil.values() for j, _ in r)
    print("cara              J/J* - 1 (5 benih)            lintasan")
    for nama, r in hasil.items():
        rel = " ".join(f"{j / J_min - 1:5.3f}" for j, _ in r)
        lin = np.median([l for _, l in r])
        print(f"{nama:16s} {rel}   "
              + ("   -" if np.isnan(lin) else f"{lin:5.2f}"))
    mb = MiniBatchKMeans(50, batch_size=1024, n_init=1, random_state=0,
                         max_no_improvement=None).fit(X)
    print("mini-batch 1024, benih 0, tanpa henti dini:")
    print(f"  J/J* - 1 = {-mb.score(X) / J_min - 1:.3f}, "
          f"{mb.n_steps_ * 1024 / len(X):.0f} lintasan")
