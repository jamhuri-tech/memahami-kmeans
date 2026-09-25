# Sumber data

Setiap data yang dipakai buku ini dicatat di sini: nama, alamat
unduhan, tanggal pengambilan, dan berkas kode yang mengunduhnya.

Data lain di buku ini dibangkitkan sendiri atau disertakan dalam
scikit-learn (`load_digits`, `load_iris`, `load_wine`,
`load_sample_image`), sehingga
tidak perlu diunduh terpisah.

## Online Retail (Bab 19)

| | |
|---|---|
| Nama | Online Retail, UCI Machine Learning Repository, id 352 |
| Rujukan | Chen, D. (2015). Online Retail [Data set]. https://doi.org/10.24432/C5BW33 |
| Lisensi | CC BY 4.0 |
| Alamat unduhan | https://archive.ics.uci.edu/static/public/352/online+retail.zip |
| Diambil | 25 September 2026 |
| Berkas | `online+retail.zip`, 23.715.478 bait, berisi `Online Retail.xlsx` |
| SHA-256 | `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b` |
| Diunduh oleh | `kode/bab19_data.py` (fungsi `baca`), sekali, lalu disimpan sebagai `data/online_retail.csv.gz` |
| Isi | 541.909 baris transaksi, 1 Desember 2010 sampai 9 Desember 2011 |

Berkas zip dan csv.gz tidak dimasukkan ke repositori; `bab19_data.py`
mengunduhnya dan memeriksa sidiknya.
