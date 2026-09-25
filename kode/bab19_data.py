"""Bab 19: data Online Retail (UCI 352) dan ringkasan RFM per pelanggan.

Berkas online+retail.zip (23,7 MB) diunduh sekali ke folder data/,
lalu dibaca dengan pandas (perlu openpyxl) dan disimpan sebagai
data/online_retail.csv.gz supaya pembacaan berikutnya cepat. Sumber dan
sidik SHA-256 dicatat di data/SUMBER.md.

bersihkan dan rfm dicetak di naskah (Listing 19.1 dan 19.2).
Dijalankan sendiri, program ini mencetak ringkasan pembersihan.
"""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
ALAMAT = ("https://archive.ics.uci.edu/static/public/352/"
          "online+retail.zip")
SHA256 = ("f5385cbb54bbebf7196389109c6b0621"
          "faab0c304e3702548165e71c84aede8b")
ACUAN = pd.Timestamp("2011-12-10")   # sehari sesudah transaksi terakhir
# kode barang yang bukan barang: ongkos kirim, potongan, biaya bank, ...
BUKAN_BARANG = ["POST", "DOT", "M", "m", "C2", "D", "S", "B",
                "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS"]


def baca():
    """Seluruh 541.909 baris transaksi, apa adanya."""
    csv = DATA / "online_retail.csv.gz"
    if not csv.exists():
        zp = DATA / "online_retail.zip"
        if not zp.exists():
            DATA.mkdir(exist_ok=True)
            urllib.request.urlretrieve(ALAMAT, zp)
        if hashlib.sha256(zp.read_bytes()).hexdigest() != SHA256:
            raise ValueError(f"{zp} tidak sama dengan versi di buku")
        with zipfile.ZipFile(zp) as z, z.open("Online Retail.xlsx") as f:
            df = pd.read_excel(f, dtype={"InvoiceNo": str,
                                         "StockCode": str})
        df.to_csv(csv, index=False)
    return pd.read_csv(csv, dtype={"InvoiceNo": str, "StockCode": str},
                       parse_dates=["InvoiceDate"])


def bersihkan(df):
    kenal = df.CustomerID.notna()
    barang = ~df.StockCode.isin(BUKAN_BARANG)
    df = df[kenal & barang & (df.UnitPrice > 0)]
    df = df.assign(nilai=df.Quantity * df.UnitPrice)
    batal = df.InvoiceNo.str.startswith("C")
    return df[~batal & (df.Quantity > 0)], df[batal]


def rfm(beli, batal, acuan=ACUAN):
    beli, batal = beli[beli.InvoiceDate < acuan], \
        batal[batal.InvoiceDate < acuan]
    g = beli.groupby("CustomerID")
    tabel = pd.DataFrame({
        "R": (acuan - g.InvoiceDate.max()).dt.days,   # hari
        "F": g.InvoiceNo.nunique(),                   # faktur
        "M": g.nilai.sum()})                          # pound
    # pembatalan bernilai negatif: M menjadi belanja bersih
    kurang = batal.groupby("CustomerID").nilai.sum()
    tabel["M"] = tabel.M.add(kurang, fill_value=0).round(2)
    return tabel[tabel.M > 0]


def fitur_log(tabel):
    """log(1 + R), log F, log M, lalu dibakukan."""
    L = np.column_stack([np.log1p(tabel.R), np.log(tabel.F),
                         np.log(tabel.M)])
    return (L - L.mean(axis=0)) / L.std(axis=0)


if __name__ == "__main__":
    df = baca()
    print(f"{len(df)} baris, {df.CustomerID.nunique()} pelanggan, "
          f"{df.InvoiceDate.min():%d-%m-%Y} sampai "
          f"{df.InvoiceDate.max():%d-%m-%Y}")
    langkah = [
        ("tanpa CustomerID", df.CustomerID.isna()),
        ("kode bukan barang", df.StockCode.isin(BUKAN_BARANG)),
        ("harga <= 0", df.UnitPrice <= 0),
        ("faktur pembatalan (C...)", df.InvoiceNo.str.startswith("C")),
        ("jumlah <= 0 bukan pembatalan",
         (df.Quantity <= 0) & ~df.InvoiceNo.str.startswith("C"))]
    for nama, m in langkah:
        print(f"  {nama:30s} {m.sum():7d} baris")
    beli, batal = bersihkan(df)
    print(f"pembelian bersih: {len(beli)} baris, "
          f"{beli.InvoiceNo.nunique()} faktur, "
          f"{beli.CustomerID.nunique()} pelanggan")
    print(f"pembatalan pelanggan: {len(batal)} baris, "
          f"nilai {batal.nilai.sum():.0f} pound")

    print()
    kotor = beli.groupby("CustomerID").nilai.sum()
    bersih = kotor.add(batal.groupby("CustomerID").nilai.sum(),
                       fill_value=0)
    print("pelanggan    M kotor    M bersih   peringkat kotor/bersih")
    pk = kotor.rank(ascending=False).astype(int)
    pb = bersih.rank(ascending=False).astype(int)
    for c in kotor.sort_values().index[-5:][::-1]:
        print(f"{c:9.0f} {kotor[c]:10.0f} {bersih[c]:11.0f}"
              f"   {pk[c]:6d} / {pb[c]}")
    tabel = rfm(beli, batal)
    print(f"RFM: {len(tabel)} pelanggan "
          f"({beli.CustomerID.nunique() - len(tabel)} dengan M bersih "
          f"<= 0 dibuang)")
