"""Bab 18: kuantisasi blok gambar abu-abu dan product quantization.

blok dan kuantisasi_produk dicetak di naskah (Listing 18.3). Gambar
china.jpg dan flower.jpg dari scikit-learn diubah menjadi abu-abu,
dipotong menjadi blok h x w, dan setiap blok menjadi satu vektor.
Kuantisasi skalar, kuantisasi vektor blok 2 x 2 dan 4 x 4, dan
product quantization dibandingkan pada laju bit per piksel yang sama.
Lalu codebook dari satu gambar dipakai untuk gambar yang lain.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

from bab04_data import BENIH
from bab18_warna import psnr


def abu(nama):
    rgb = load_sample_image(nama).astype(float)
    return rgb @ [0.299, 0.587, 0.114]     # luminans


def blok(g, h, w):
    """Blok h x w dari gambar g; satu baris per blok."""
    H, W = g.shape[0] // h * h, g.shape[1] // w * w
    B = g[:H, :W].reshape(H // h, h, W // w, w)
    return B.transpose(0, 2, 1, 3).reshape(-1, h * w)


def kuantisasi_produk(X, m, K):
    """Potong setiap vektor menjadi m bagian; satu K-means per
    bagian. Kode setiap vektor: m indeks, m * log2(K) bit."""
    bagian = np.split(np.arange(X.shape[1]), m)
    model = []
    for j in bagian:
        km = KMeans(K, n_init=1, random_state=BENIH)
        model.append(km.fit(X[:, j]))
    return model, bagian


def mse_vq(km, X):
    return -km.score(X) / X.size


def mse_pq(model, bagian, X):
    return sum(-km.score(X[:, j]) for km, j in
               zip(model, bagian)) / X.size


if __name__ == "__main__":
    g = abu("china.jpg")
    print("china.jpg abu-abu, codebook dilatih pada gambar itu sendiri")
    print("cara                  bit/piksel   ukuran codebook   PSNR")
    for nama, h, w, K, m in [("skalar, K = 2", 1, 1, 2, 1),
                             ("skalar, K = 4", 1, 1, 4, 1),
                             ("blok 2x2, K = 16", 2, 2, 16, 1),
                             ("blok 2x2, K = 256", 2, 2, 256, 1),
                             ("blok 4x4, K = 256", 4, 4, 256, 1),
                             ("blok 4x4, PQ 2 x 16", 4, 4, 16, 2),
                             ("blok 4x4, PQ 2 x 256", 4, 4, 256, 2)]:
        X = blok(g, h, w)
        model, bagian = kuantisasi_produk(X, m, K)
        bpp = m * np.log2(K) / (h * w)
        print(f"{nama:22s} {bpp:6.2f} {m * K:13d} x {h * w // m:<2d}"
              f"  {psnr(mse_pq(model, bagian, X)):7.2f}")

    print()
    f = abu("flower.jpg")
    print("PSNR gambar (baris) dengan codebook dari gambar (kolom)")
    print("blok    K  gambar     china   flower")
    for h, K in ((1, 4), (2, 16), (2, 256), (4, 256)):
        Xc, Xf = blok(g, h, h), blok(f, h, h)
        kc = KMeans(K, n_init=1, random_state=BENIH).fit(Xc)
        kf = KMeans(K, n_init=1, random_state=BENIH).fit(Xf)
        print(f"{h}x{h}  {K:4d}  china    {psnr(mse_vq(kc, Xc)):7.2f}"
              f"  {psnr(mse_vq(kf, Xc)):7.2f}")
        print(f"{'':11s}flower   {psnr(mse_vq(kc, Xf)):7.2f}"
              f"  {psnr(mse_vq(kf, Xf)):7.2f}")
