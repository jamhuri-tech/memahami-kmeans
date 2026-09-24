"""Bab 12: EM berhenti di dataran jika tol terlalu longgar.

Data lonjong, kovarians penuh, n_init = 5, beberapa nilai tol.
"""
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

from bab04_data import BENIH
from bab11_data import enam_data

if __name__ == "__main__":
    _, X, y = enam_data()[1]
    print("lonjong, kovarians penuh:")
    print("     tol  n_iter    ARI   rata log p(x)")
    for tol in (1e-3, 1e-4, 1e-6, 1e-8):
        gm = GaussianMixture(3, tol=tol, max_iter=1000, n_init=5,
                             random_state=BENIH).fit(X)
        print(f"  {tol:6.0e} {gm.n_iter_:7d} {ari(y, gm.predict(X)):6.3f}"
              f" {gm.lower_bound_:14.4f}")
