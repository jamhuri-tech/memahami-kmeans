"""Bab 6: metode Hartigan, memindahkan titik satu per satu.

Fungsi hartigan dicetak di naskah (Listing 6.2). Setiap pemindahan
memakai perubahan inersia yang tepat, termasuk pergeseran kedua
centroid yang terlibat.
"""
import numpy as np

from bab04_data import BENIH, NAMA, delapan_titik, gumpalan
from bab05_lloyd import lloyd


def hartigan(X, label, K, maks_putaran=100):
    label = label.copy()
    nk = np.bincount(label, minlength=K).astype(float)
    mu = np.array([X[label == k].mean(axis=0)
                   for k in range(K)])
    for putaran in range(maks_putaran):
        pindah = 0
        for i in range(len(X)):
            a = label[i]
            if nk[a] == 1:          # jangan kosongkan klaster
                continue
            d = ((X[i] - mu) ** 2).sum(axis=1)
            biaya = nk / (nk + 1) * d
            biaya[a] = nk[a] / (nk[a] - 1) * d[a]
            b = biaya.argmin()
            if b != a:                  # pindah menurunkan J
                mu[a] = (nk[a] * mu[a] - X[i]) / (nk[a] - 1)
                mu[b] = (nk[b] * mu[b] + X[i]) / (nk[b] + 1)
                nk[a] -= 1
                nk[b] += 1
                label[i] = b
                pindah += 1
        if pindah == 0:
            break
    return label, putaran + 1


def inersia_label(X, label, K):
    mu = np.array([X[label == k].mean(axis=0) for k in range(K)])
    return ((X - mu[label]) ** 2).sum()


def teks(label, K):
    return " | ".join("".join(NAMA[i] for i in np.flatnonzero(label == k))
                      for k in range(K))


if __name__ == "__main__":
    X = delapan_titik()
    for awal in ([0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]):
        awal = np.array(awal)
        akhir, p = hartigan(X, awal, 2)
        print(f"{teks(awal, 2):10s} J = {inersia_label(X, awal, 2):6.2f}"
              f"  ->  {teks(akhir, 2):10s} "
              f"J = {inersia_label(X, akhir, 2):6.2f}")

    Xg, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    J_l, J_lh = [], []
    for _ in range(1000):
        C0 = Xg[rng.choice(len(Xg), 5, replace=False)]
        _, label, _ = lloyd(Xg, C0)
        J_l.append(inersia_label(Xg, label, 5))
        label_h, _ = hartigan(Xg, label, 5)
        J_lh.append(inersia_label(Xg, label_h, 5))
    J_l, J_lh = np.array(J_l), np.array(J_lh)
    terbaik = min(J_l.min(), J_lh.min())
    print("gumpalan, 1000 awal acak:")
    print(f"  Hartigan menurunkan J hasil Lloyd pada "
          f"{(J_lh < J_l - 1e-9).sum()} awal")
    for nama, J in (("Lloyd", J_l), ("Lloyd lalu Hartigan", J_lh)):
        print(f"  {nama:20s} median {np.median(J):8.2f}  "
              f"> 1.1 x terbaik: {(J > 1.1 * terbaik).sum():3d}")
