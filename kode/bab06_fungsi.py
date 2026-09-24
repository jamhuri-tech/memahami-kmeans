"""Bab 6: inersia sebagai fungsi centroid saja, beserta turunannya.

F(mu) = sum_i min_k ||x_i - mu_k||^2. Di luar titik seri, F mulus
sepotong-sepotong: gradien 2 n_k (mu_k - rata-rata klaster k) dan
Hessian 2 n_k I (Listing 6.1).
"""
import numpy as np
from bab05_lloyd import jarak2


def tiga_kelompok(benih=20260924):
    """Enam puluh titik di garis bilangan, tiga kelompok rapat."""
    rng = np.random.default_rng(benih)
    x = np.concatenate([rng.normal(0.0, 0.5, 25),
                        rng.normal(4.5, 0.5, 15),
                        rng.normal(9.0, 0.5, 20)])
    return np.sort(x).reshape(-1, 1)


def F(X, C):
    return jarak2(X, C).min(axis=1).sum()


def gradien_hessian(X, C):
    K, d = C.shape
    label = jarak2(X, C).argmin(axis=1)
    nk = np.bincount(label, minlength=K)
    jumlah = np.zeros((K, d))
    np.add.at(jumlah, label, X)
    g = 2 * (nk[:, None] * C - jumlah)     # gradien
    h = 2 * nk                             # Hessian 2 n_k I
    return g, h


def langkah_newton(X, C):
    g, h = gradien_hessian(X, C)
    return C - g / h[:, None]
