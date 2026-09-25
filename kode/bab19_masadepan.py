"""Bab 19: apakah segmen meramalkan belanja berikutnya?

RFM dan segmen dihitung hanya dari transaksi sebelum 1 September 2011.
Untuk setiap segmen dicatat bagian pelanggan yang membeli lagi antara 1
September dan 9 Desember 2011, serta rata-rata belanjanya. Pembanding:
skor RFM kuartil yang lazim dalam pemasaran. Ukuran ringkas: R^2
log(1 + belanja berikutnya) yang dijelaskan oleh rata-rata segmen.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from bab04_data import BENIH
from bab19_data import baca, bersihkan, fitur_log, rfm
from bab19_segmen import segmen

POTONG = pd.Timestamp("2011-09-01")


def skor_kuartil(tabel):
    """Skor 0..9: kuartil R (terbalik), F, dan M dijumlahkan, lalu
    dipotong menjadi empat kelompok."""
    def q(s, naik):
        return pd.qcut(s.rank(method="first", ascending=naik), 4,
                       labels=False)
    skor = q(tabel.R, False) + q(tabel.F, True) + q(tabel.M, True)
    return pd.cut(skor, [-1, 2, 4, 6, 9], labels=False).to_numpy()


def r2(y, label):
    m = pd.Series(y).groupby(label).transform("mean").to_numpy()
    return 1 - ((y - m) ** 2).sum() / ((y - y.mean()) ** 2).sum()


if __name__ == "__main__":
    beli, batal = bersihkan(baca())
    tabel = rfm(beli, batal, acuan=POTONG)
    nanti = beli[beli.InvoiceDate >= POTONG].groupby(
        "CustomerID").nilai.sum()
    tabel["belanja"] = nanti.reindex(tabel.index).fillna(0)
    print(f"{len(tabel)} pelanggan sebelum 1 September 2011; "
          f"{(tabel.belanja > 0).mean():.3f} membeli lagi")
    y = np.log1p(tabel.belanja.to_numpy())

    label, _ = segmen(tabel)
    grup = skor_kuartil(tabel)
    for nama, lab in (("segmen K-means", label),
                      ("kelompok skor kuartil", grup)):
        print()
        print(f"{nama}:")
        print("  no     n  med R  med F  med M  membeli lagi  belanja")
        for k in range(4):
            t = tabel[lab == k]
            print(f"{k + 1:4d} {len(t):5d} {t.R.median():6.0f} "
                  f"{t.F.median():6.0f} {t.M.median():6.0f}"
                  f"  {(t.belanja > 0).mean():12.3f}"
                  f"  {t.belanja.mean():7.0f}")
    print()
    print(f"ARI segmen lawan skor kuartil: "
          f"{adjusted_rand_score(label, grup):.3f}")
    print("R^2 log(1 + belanja berikutnya)")
    print(f"  skor kuartil         {r2(y, grup):.3f}")
    Z = fitur_log(tabel)
    for K in (2, 3, 4, 5, 6, 8, 10):
        km = KMeans(K, n_init=10, random_state=BENIH).fit(Z)
        print(f"  K-means, K = {K:2d}      {r2(y, km.labels_):.3f}")
