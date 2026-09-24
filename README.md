# Memahami K-Means: kode pendamping

Kode Python untuk buku *Memahami K-Means* karya Mohammad Jamhuri. Setiap angka dan gambar di buku berasal dari menjalankan berkas di folder `kode/` dan `gen_gambar.py`.

## Menjalankan

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd kode
../.venv/bin/python bab05_tangan.py
```

Program dijalankan dari dalam folder `kode/`, karena setiap berkas mengimpor berkas bab sebelumnya. Semua gambar dapat dibangkitkan ulang dengan `python gen_gambar.py` dari folder utama (hasilnya di `gbr/`).

Versi yang dipakai buku: Python 3.9.6 dengan pustaka di `requirements.txt`. Setiap pembangkit bilangan acak memakai benih tetap, sehingga keluaran dapat diulang. Versi lain biasanya memberi angka yang sama sampai beberapa digit, tetapi tidak dijamin sama persis.

## Isi per bab

### Bab 4 - Masalah K-Means: Inersia dan Partisi

- `bab04_data.py`: data yang dipakai sepanjang Bagian II.
- `bab04_inersia.py`: centroid dan inersia sebuah partisi.
- `bab04_rata.py`: tiga identitas jumlah kuadrat, diperiksa dengan angka.
- `bab04_satudimensi.py`: K-means satu dimensi yang tepat, dengan pemrograman dinamis.
- `bab04_semua.py`: menelusuri semua partisi delapan titik.

### Bab 5 - Algoritma Lloyd Langkah demi Langkah

- `bab05_iterasi.py`: berapa iterasi yang diperlukan algoritma Lloyd?
- `bab05_kosong.py`: seri jarak dan klaster kosong.
- `bab05_lloyd.py`: algoritma Lloyd dalam bentuk paling sederhana.
- `bab05_lokal.py`: ke titik tetap mana algoritma Lloyd berhenti?
- `bab05_monoton.py`: inersia tidak pernah naik, diperiksa pada 200 awal acak.
- `bab05_tangan.py`: algoritma Lloyd pada delapan titik, dengan tangan.

### Bab 6 - K-Means sebagai Optimisasi

- `bab06_fungsi.py`: inersia sebagai fungsi centroid saja, beserta turunannya.
- `bab06_hartigan.py`: metode Hartigan, memindahkan titik satu per satu.
- `bab06_lanskap.py`: lanskap F(mu1, mu2) pada data satu dimensi, K = 2.
- `bab06_newton.py`: gradien, Hessian, dan langkah Newton pada data gumpalan.

### Bab 7 - K-Means dari Nol

- `bab07_cocok.py`: KMeansKita lawan KMeans scikit-learn, angka demi angka.
- `bab07_kmeans.py`: kelas KMeansKita, tiruan KMeans(algorithm="lloyd").
- `bab07_presisi.py`: mengapa scikit-learn memusatkan data sebelum menghitung jarak.
- `bab07_seri.py`: seri tepat pada data bulat (digits) dan akibatnya.
- `bab07_tol.py`: toleransi relatif tol dan penugasan akhir.

### Bab 8 - Inisialisasi dan k-means++

- `bab08_awal.py`: lima cara memilih centroid awal, 1000 awal masing-masing.
- `bab08_lain.py`: dua pemeriksaan tambahan.
- `bab08_ninit.py`: berapa banyak awal yang perlu dicoba (n_init)?
- `bab08_plusplus.py`: k-means++ dari nol, serakah seperti scikit-learn.

### Bab 9 - Memilih Banyaknya Klaster

- `bab09_data.py`: empat data untuk memilih K.
- `bab09_gap.py`: gap statistic (Tibshirani, Walther, dan Hastie 2001).
- `bab09_pilih.py`: K yang dipilih setiap ukuran pada keempat data.
- `bab09_siku.py`: kurva inersia J(K) untuk K = 1..10 pada keempat data.
- `bab09_ukuran.py`: silhouette, Calinski-Harabasz, dan Davies-Bouldin dari nol.

### Bab 10 - Menilai Hasil Clustering

- `bab10_banding.py`: memakai ukuran eksternal.
- `bab10_cocok.py`: ukuran eksternal kita lawan sklearn.metrics.
- `bab10_digits.py`: K-means pada digits dibandingkan dengan label angka.
- `bab10_kebetulan.py`: ukuran eksternal untuk partisi acak terhadap label digits.
- `bab10_ukuran.py`: ukuran eksternal dari tabel kontingensi.

### Bab 11 - Asumsi Tersembunyi dan Penskalaan

- `bab11_banding.py`: K-means dibandingkan dengan Ward dan DBSCAN.
- `bab11_data.py`: enam data berlabel untuk menguji anggapan K-means.
- `bab11_gagal.py`: K-means pada enam data berlabel.
- `bab11_pemutih.py`: PCA dan pemutihan (whitening) sebelum K-means.
- `bab11_skala.py`: penskalaan peubah dapat memperbaiki atau merusak K-means.

### Bab 12 - Dari K-Means ke Campuran Gaussian

- `bab12_batas.py`: K-means sebagai batas campuran Gaussian dengan sigma -> 0.
- `bab12_cocok.py`: CampuranKita lawan GaussianMixture, lima data, benih 0..4.
- `bab12_dataran.py`: EM berhenti di dataran jika tol terlalu longgar.
- `bab12_em.py`: EM untuk campuran Gaussian berkovarians penuh, dari nol.
- `bab12_jenis.py`: empat jenis kovarians lawan K-means pada enam data Bab 11.
- `bab12_runtuh.py`: likelihood yang tak terbatas.

### Bab 13 - Pencilan dan Jarak Lain

- `bab13_bola.py`: spherical K-means untuk data yang klasternya berupa arah.
- `bab13_bregman.py`: pusat terbaik untuk berbagai ukuran ketakmiripan.
- `bab13_kokoh.py`: tiga varian K-means yang kokoh terhadap pencilan.
- `bab13_pencilan.py`: empat metode pada data gumpalan dengan m pencilan.

### Bab 14 - Kernel K-Means dan Spectral Clustering

- `bab14_angkat.py`: membawa data cincin ke ruang fitur secara eksplisit.
- `bab14_kernel.py`: kernel K-means dari nol.
- `bab14_percobaan.py`: kernel K-means pada bulan, cincin, dan gumpalan.
- `bab14_relaksasi.py`: kernel K-means sebagai masalah nilai eigen yang dikekang.
- `bab14_spektral.py`: spectral clustering dari nol (Ng, Jordan, dan Weiss).

## Lisensi

Kode boleh dipakai, disalin, diubah, dan disebarluaskan secara bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban mencantumkan sumber (lisensi MIT-0). Teks buku tidak termasuk dalam lisensi ini.

Saran dan koreksi: m.jamhuri@live.com
