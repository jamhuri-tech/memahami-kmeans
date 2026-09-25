"""Bab 20: HDBSCAN pada enam data Bab 11 dan pada fitur log RFM Bab 19.

HDBSCAN tidak memerlukan eps; parameter utamanya min_cluster_size.
Untuk setiap data dicetak ARI, banyaknya klaster, dan bagian titik yang
dianggap noise. Pada RFM, klaster HDBSCAN dibandingkan dengan nilai F,
lalu F digoyang seragam +-0,5 sebelum log untuk menghapus tumpukan
pada bilangan bulat.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from bab04_data import BENIH
from bab11_data import enam_data
from bab19_data import baca, bersihkan, fitur_log, rfm


def ringkas(label):
    k = label.max() + 1
    return k, (label < 0).mean()


if __name__ == "__main__":
    print(f"{'':9s}" + "".join(f"{'min_cluster_size = ' + str(m):>22s}"
                             for m in (10, 40)))
    print(f"{'data':9s}" + " {:>6s} {:>7s} {:>6s}".format(
        "ARI", "klaster", "noise") * 2)
    for nama, X, y in enam_data():
        baris = f"{nama:9s}"
        for m in (10, 40):
            lab = HDBSCAN(min_cluster_size=m).fit(X).labels_
            k, noise = ringkas(lab)
            baris += (f" {adjusted_rand_score(y, lab):6.3f} {k:7d}"
                      f" {noise:6.3f}")
        print(baris)

    print()
    tabel = rfm(*bersihkan(baca()))
    lab = HDBSCAN(min_cluster_size=50).fit(fitur_log(tabel)).labels_
    print("RFM, min_cluster_size = 50: banyaknya pelanggan")
    silang = pd.crosstab(lab, np.minimum(tabel.F, 5).to_numpy())
    print("klaster    F=1   F=2   F=3   F=4  F>=5")
    for k, baris in silang.iterrows():
        nama = "noise" if k < 0 else str(k)
        print(f"{nama:7s}" + "".join(f"{v:6d}" for v in baris))

    rng = np.random.default_rng(BENIH)
    goyang = tabel.assign(F=tabel.F + rng.uniform(-0.5, 0.5, len(tabel)))
    Z = fitur_log(goyang)
    print("F digoyang +-0,5:")
    for m in (20, 50, 100):
        k, noise = ringkas(HDBSCAN(min_cluster_size=m).fit(Z).labels_)
        print(f"  min_cluster_size = {m:3d}: {k} klaster, "
              f"noise {noise:.3f}")
