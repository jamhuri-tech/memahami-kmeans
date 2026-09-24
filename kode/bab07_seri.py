"""Bab 7: seri tepat pada data bulat (digits) dan akibatnya.

Centroid awal "random" berupa titik data, sehingga pada iterasi
pertama jarak kuadrat berupa bilangan bulat dan seri tepat mungkin.
Rumus ||c||^2 - 2 x.c pada data terpusat memutus seri itu dengan
galat pembulatan, yang berbeda di NumPy dan di BLAS scikit-learn.
"""
import numpy as np
from sklearn.cluster._kmeans import _kmeans_single_lloyd, _tolerance
from sklearn.datasets import load_digits

from bab07_kmeans import lloyd_satu

if __name__ == "__main__":
    X = load_digits().data.astype(float)
    n, K = len(X), 10
    w = np.ones(n)
    Xc = X - X.mean(axis=0)
    tol = _tolerance(X, 1e-4)
    jalan = seri = beda = 0
    for s in range(10):
        rng = np.random.RandomState(s)
        for r in range(10):
            idx = rng.choice(n, K, replace=False, p=w / w.sum())
            D = ((X[:, None, :] - X[idx][None]) ** 2).sum(axis=2)
            urut = np.sort(D, axis=1)
            ada_seri = (urut[:, 0] == urut[:, 1]).any()
            a = lloyd_satu(Xc, w, Xc[idx].copy(), 300, tol)
            b = _kmeans_single_lloyd(Xc, w, Xc[idx].copy(),
                                     max_iter=300, tol=tol)
            sama = np.array_equal(a[0], b[0])
            jalan += 1
            seri += ada_seri
            beda += not sama
            if not sama and not ada_seri:
                print(f"  beda tanpa seri: benih {s}, jalan {r}")
    print(f"digits, {jalan} jalan Lloyd (10 benih x 10 awal):")
    print(f"  jalan dengan seri tepat di iterasi 1: {seri}")
    print(f"  jalan yang labelnya berbeda dari scikit-learn: {beda}")
