"""Bab 3: benih acak.

np.random.default_rng (Generator, dipakai kode kita) dan
np.random.RandomState (dipakai scikit-learn) memberi deret yang berbeda
dari benih yang sama. make_blobs dengan random_state tetap memberi data
yang sama setiap kali.
"""
import numpy as np
from sklearn.datasets import make_blobs

from bab04_data import BENIH

if __name__ == "__main__":
    print("default_rng(0).integers(0, 100, 5):",
          np.random.default_rng(0).integers(0, 100, 5).tolist())
    print("RandomState(0).randint(0, 100, 5): ",
          np.random.RandomState(0).randint(0, 100, 5).tolist())
    print("RandomState(0) sekali lagi:        ",
          np.random.RandomState(0).randint(0, 100, 5).tolist())
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 100, 3), rng.integers(0, 100, 3)
    print("satu Generator, dua panggilan:", a.tolist(), b.tolist())

    X1, y1 = make_blobs(300, centers=3, random_state=BENIH)
    X2, y2 = make_blobs(300, centers=3, random_state=BENIH)
    X3, _ = make_blobs(300, centers=3, random_state=BENIH + 1)
    print(f"make_blobs, benih sama : sama = {np.array_equal(X1, X2)}")
    print(f"make_blobs, benih beda : sama = {np.array_equal(X1, X3)}")
    print(f"X1 {X1.shape}, y1 {np.bincount(y1).tolist()}")
