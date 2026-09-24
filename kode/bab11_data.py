"""Bab 11: enam data berlabel untuk menguji anggapan K-means.

gumpalan  : acuan, tiga gumpalan bulat sama besar dan sama sebaran
lonjong   : tiga gumpalan yang ditarik miring oleh satu peta linear
timpang   : satu gumpalan besar (500 titik) dan dua kecil (25 titik)
sebaran   : tiga gumpalan dengan simpangan baku 0,5; 1,5; dan 3
bulan     : dua bulan sabit (make_moons)
cincin    : dua lingkaran sepusat (make_circles)
"""
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

from bab04_data import BENIH


def enam_data():
    rng = np.random.default_rng(BENIH)
    X0, y0 = make_blobs(600, centers=[[0, 0], [5, 0], [2.5, 4]],
                        cluster_std=1.0, random_state=BENIH)
    Xl, yl = make_blobs(600, centers=[[0, 0], [3, 3], [6, 6]],
                        cluster_std=1.0, random_state=BENIH)
    Xl = Xl @ np.array([[0.6, -0.64], [-0.4, 0.85]])
    pusat = np.array([[0, 0], [4.5, 0], [4.5, 3.2]])
    ukuran = [500, 25, 25]
    Xt = np.vstack([c + rng.normal(size=(m, 2))
                    for c, m in zip(pusat, ukuran)])
    yt = np.repeat([0, 1, 2], ukuran)
    Xs, ys = make_blobs(600, centers=[[0, 0], [4, 0], [10, 1]],
                        cluster_std=[0.5, 1.5, 3.0], random_state=BENIH)
    Xb, yb = make_moons(600, noise=0.06, random_state=BENIH)
    Xc, yc = make_circles(600, noise=0.05, factor=0.45,
                          random_state=BENIH)
    return [("gumpalan", X0, y0), ("lonjong", Xl, yl),
            ("timpang", Xt, yt), ("sebaran", Xs, ys),
            ("bulan", Xb, yb), ("cincin", Xc, yc)]
