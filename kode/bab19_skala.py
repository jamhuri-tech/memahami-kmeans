"""Bab 19: sebaran R, F, M dan akibat penskalaan pada K-means, K = 4.

Tiga pilihan: nilai mentah, nilai dibakukan, dan log lalu dibakukan
(fitur_log). Untuk setiap pilihan dicetak ukuran keempat klaster.
"""
import numpy as np
from scipy.stats import skew
from sklearn.cluster import KMeans

from bab04_data import BENIH
from bab19_data import baca, bersihkan, fitur_log, rfm

if __name__ == "__main__":
    tabel = rfm(*bersihkan(baca()))
    X = tabel[["R", "F", "M"]].to_numpy(float)
    print("   " + "".join(f"{h:>9s}" for h in ("min", "kuartil1",
          "median", "kuartil3", "maks")) + "  kemencengan")
    for j, v in enumerate("RFM"):
        q = np.percentile(X[:, j], [0, 25, 50, 75, 100])
        print(f"{v}  " + "".join(f"{a:9.0f}" for a in q) +
              f"  {skew(X[:, j]):11.2f}")
    Z = fitur_log(tabel)
    print("kemencengan sesudah log:",
          " ".join(f"{skew(Z[:, j]):.2f}" for j in range(3)))
    print("korelasi log:  R-F %.2f   R-M %.2f   F-M %.2f" % tuple(
        np.corrcoef(Z.T)[[0, 0, 1], [1, 2, 2]]))

    print()
    print("penskalaan       ukuran keempat klaster, K = 4")
    for nama, A in (("mentah", X),
                    ("dibakukan", (X - X.mean(0)) / X.std(0)),
                    ("log, dibakukan", Z)):
        km = KMeans(4, n_init=10, random_state=BENIH).fit(A)
        print(f"{nama:16s} {sorted(np.bincount(km.labels_).tolist())}")
