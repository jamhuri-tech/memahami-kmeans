"""Bab 12: CampuranKita lawan GaussianMixture, lima data, benih 0..4."""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from bab11_data import enam_data
from bab12_em import CampuranKita


def angka(v, batas=1e-12):
    return f"{'< ' + format(batas, '.0e'):>9s}" if v < batas \
        else f"{v:9.1e}"


if __name__ == "__main__":
    data = [(n, X, y.max() + 1) for n, X, y in enam_data()[:4]]
    data.append(("wine baku", StandardScaler().fit_transform(
        load_wine().data), 3))
    print("data        n_iter  |d mu|    |d S|  |d batas|")
    for nama, X, K in data:
        it, dm, dS, db = 0, 0, 0, 0
        for s in range(5):
            a = CampuranKita(K, random_state=s).fit(X)
            b = GaussianMixture(K, random_state=s).fit(X)
            it += a.n_iter_ == b.n_iter_
            dm = max(dm, np.abs(a.means_ - b.means_).max())
            dS = max(dS, np.abs(a.covariances_ - b.covariances_).max())
            db = max(db, abs(a.lower_bound_ - b.lower_bound_))
        print(f"{nama:10s} {it:3d}/5 {angka(dm)} {angka(dS)} "
              f"{angka(db)}")
