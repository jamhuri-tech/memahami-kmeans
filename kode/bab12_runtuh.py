"""Bab 12: likelihood yang tak terbatas.

Data gumpalan ditambah empat salinan satu titik yang jauh. Satu
komponen dapat menempel pada titik kembar itu, dan kovariansnya menyusut
sampai reg_covar, sehingga likelihood naik tanpa batas ketika reg_covar
makin kecil.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH, gumpalan

if __name__ == "__main__":
    X, _ = gumpalan()
    X = np.vstack([X, np.tile([[15.0, 12.0]], (4, 1))])
    print("  reg_covar   rata log p(x)   det kovarians terkecil")
    for r in (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        gm = GaussianMixture(6, reg_covar=r, n_init=3,
                             random_state=BENIH).fit(X)
        det = np.linalg.det(gm.covariances_).min()
        print(f"  {r:9.0e} {gm.lower_bound_:15.4f} {det:24.2e}")
    try:
        gm = GaussianMixture(6, reg_covar=0, n_init=3,
                             random_state=BENIH).fit(X)
        print(f"  reg_covar=0: rata log p(x) {gm.lower_bound_:.4f}")
    except ValueError as e:
        print("  reg_covar=0: ValueError:", str(e)[:36], "...")
    print(f"  kenaikan per faktor 100, 4 titik kembar dari {len(X)}: "
          f"{4 * np.log(100) / len(X):.4f}")
