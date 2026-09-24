"""Bab 5: algoritma Lloyd dalam bentuk paling sederhana.

Fungsi lloyd dicetak di naskah (Listing 5.1). Fungsi ini sengaja
polos: tanpa toleransi, tanpa penanganan klaster kosong. Bab 7
menulis versi lengkapnya.
"""
import numpy as np


def jarak2(X, C):
    """Kuadrat jarak setiap titik ke setiap centroid."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def lloyd(X, C, maks_iter=100):
    K = len(C)
    riwayat = []
    lama = None
    for t in range(maks_iter):
        D = jarak2(X, C)
        label = D.argmin(axis=1)                  # penugasan
        J_tugas = D[np.arange(len(X)), label].sum()
        if lama is not None and np.array_equal(label, lama):
            break                             # label tetap
        C = np.array([X[label == k].mean(axis=0)  # pembaruan
                      for k in range(K)])
        J_baru = ((X - C[label]) ** 2).sum()
        riwayat.append((J_tugas, J_baru, C.copy(), label))
        lama = label
    return C, label, riwayat
