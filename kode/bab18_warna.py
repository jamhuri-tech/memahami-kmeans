"""Bab 18: kuantisasi warna gambar china.jpg dari scikit-learn.

kuantisasi_warna dicetak di naskah (Listing 18.2): palet K warna
dipelajari dari 10.000 piksel acak, lalu setiap piksel diganti dengan
warna palet terdekat. Dibandingkan dengan K-means pada seluruh piksel
dan dengan palet seragam (L tingkat per kanal, K = L^3).
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

from bab04_data import BENIH


def psnr(mse):
    return 10 * np.log10(255 ** 2 / mse)


def kuantisasi_warna(gambar, K, n_sampel=10_000, benih=BENIH):
    X = gambar.reshape(-1, 3).astype(float)  # baris = piksel
    rng = np.random.default_rng(benih)
    S = X[rng.choice(len(X), n_sampel, replace=False)]
    km = KMeans(K, n_init=1, random_state=benih).fit(S)
    label = km.predict(X)       # log2(K) bit per piksel
    palet = km.cluster_centers_
    return palet, label


def palet_seragam(X, L):
    """L tingkat per kanal di tengah selang yang sama lebar."""
    return (np.floor(X / 256 * L) + 0.5) * 256 / L


if __name__ == "__main__":
    gambar = load_sample_image("china.jpg")
    X = gambar.reshape(-1, 3).astype(float)
    print(f"{gambar.shape[0]} x {gambar.shape[1]} piksel, "
          f"{len(np.unique(X, axis=0))} warna berbeda")
    print("   K  bit/piksel   PSNR sampel  PSNR penuh  "
          "PSNR seragam")
    hasil = {}
    for K in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        palet, label = kuantisasi_warna(gambar, K)
        mse = ((X - palet[label]) ** 2).mean()
        km = KMeans(K, n_init=1, random_state=BENIH).fit(X)
        baris = (f"{K:4d}  {np.log2(K):10.0f}   {psnr(mse):11.2f}"
                 f"  {psnr(km.inertia_ / X.size):10.2f}")
        L = round(K ** (1 / 3))
        if L ** 3 == K:
            q = palet_seragam(X, L)
            baris += f"  {psnr(((X - q) ** 2).mean()):12.2f}"
        print(baris)
        hasil[K] = psnr(mse)
    naik = (hasil[512] - hasil[16]) / 5
    print(f"kenaikan PSNR per bit, K = 16 sampai 512: {naik:.2f} dB")

    palet, label = kuantisasi_warna(gambar, 16)
    galat = np.sqrt(((X - palet[label]) ** 2).sum(axis=1))
    print("K = 16, jarak warna asli ke warna palet")
    print("  persentil   50     99   99.9    100")
    print("  jarak   " + "".join(f"{v:7.1f}" for v in
                                 np.percentile(galat, [50, 99, 99.9, 100])))
