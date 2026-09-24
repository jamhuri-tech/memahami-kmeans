"""Bab 12: EM untuk campuran Gaussian berkovarians penuh, dari nol.

log_normal, langkah_e, dan langkah_m dicetak di naskah (Listing 12.1
dan 12.2). Kelas CampuranKita meniru GaussianMixture(
covariance_type="full", init_params="kmeans", n_init=1).
"""
import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.special import logsumexp
from sklearn.cluster import KMeans


def log_normal(X, mu, S):
    """log N(x | mu, S) untuk setiap baris X."""
    L = cholesky(S, lower=True)
    z = solve_triangular(L, (X - mu).T, lower=True)
    d = X.shape[1]
    log_det = 2 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + log_det
                   + (z ** 2).sum(axis=0))


def langkah_e(X, pi, mu, S):
    L = np.stack([np.log(pi[k]) + log_normal(X, mu[k], S[k])
                  for k in range(len(pi))], axis=1)
    log_px = logsumexp(L, axis=1)              # log p(x_i)
    return np.exp(L - log_px[:, None]), log_px.mean()


def langkah_m(X, g, reg=1e-6):
    nk = g.sum(axis=0) + 10 * np.finfo(float).eps
    mu = g.T @ X / nk[:, None]
    S = np.empty((len(nk), X.shape[1], X.shape[1]))
    for k in range(len(nk)):
        D = X - mu[k]
        S[k] = (g[:, k] * D.T) @ D / nk[k]
        S[k].flat[::X.shape[1] + 1] += reg     # ke diagonal
    return nk / len(X), mu, S


class CampuranKita:
    def __init__(self, K, tol=1e-3, max_iter=100, reg=1e-6,
                 random_state=None):
        self.K, self.tol, self.max_iter = K, tol, max_iter
        self.reg, self.random_state = reg, random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        lab = KMeans(self.K, n_init=1, random_state=rng).fit(X).labels_
        g = np.eye(self.K)[lab]                # awal: label K-means
        pi, mu, S = langkah_m(X, g, self.reg)
        batas = -np.inf
        self.riwayat = []
        for it in range(1, self.max_iter + 1):
            lama = batas
            g, batas = langkah_e(X, pi, mu, S)
            pi, mu, S = langkah_m(X, g, self.reg)
            self.riwayat.append(batas)
            if abs(batas - lama) < self.tol:
                break
        self.weights_, self.means_, self.covariances_ = pi, mu, S
        self.lower_bound_, self.n_iter_ = batas, it
        return self

    def predict_proba(self, X):
        return langkah_e(X, self.weights_, self.means_,
                         self.covariances_)[0]
