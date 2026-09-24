"""Bab 16: minibatch lawan MiniBatchKMeans.partial_fit dengan batch yang
sama, dan K-means online lawan algoritma Lloyd."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

from bab04_data import BENIH, gumpalan
from bab05_lloyd import lloyd
from bab16_mini import macqueen, minibatch


def inersia(X, C):
    return ((X[:, None] - C[None]) ** 2).sum(axis=2).min(axis=1).sum()


if __name__ == "__main__":
    X, _ = gumpalan(20000)
    C0, _ = kmeans_plusplus(X, 5, random_state=0)
    kita = minibatch(X, C0, 256, 200, np.random.RandomState(BENIH))
    mbk = MiniBatchKMeans(5, init=C0, n_init=1, reassignment_ratio=0)
    rng = np.random.RandomState(BENIH)
    for _ in range(200):
        mbk.partial_fit(X[rng.randint(0, len(X), 256)])
    dC = np.abs(kita - mbk.cluster_centers_).max()
    print("minibatch lawan partial_fit, 200 x 256 titik: "
          + ("|dC| < 1e-12" if dC < 1e-12 else f"|dC| = {dC:.1e}"))
    C_l, _, r = lloyd(X, C0)
    J_l = inersia(X, C_l)
    print(f"Lloyd: {len(r) + 1} iterasi x {len(X)} titik, "
          f"J = {J_l:.1f}")
    rng = np.random.default_rng(BENIH)
    C = C0
    for p in range(1, 4):
        C = macqueen(X, C, rng.permutation(len(X)))
        print(f"MacQueen, {p} lintasan: J = {inersia(X, C):.1f}"
              f" (selisih relatif {inersia(X, C) / J_l - 1:.1e})")
