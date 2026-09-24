"""Bab 8: berapa banyak awal yang perlu dicoba (n_init)?

Seribu jalan dari bab08_awal dibagi menjadi kelompok berisi m jalan.
Setiap kelompok meniru n_init = m: yang disimpan inersia terkecilnya.
"""
import numpy as np

from bab08_awal import CARA, jalankan, kisi

if __name__ == "__main__":
    X, K = kisi(), 25
    hasil = {nama: jalankan(X, K, f)[1] for nama, f in CARA
             if nama in ("forgy", "pp_asli", "pp_serakah")}
    terbaik = min(j.min() for j in hasil.values())
    print(f"kisi 5x5: bagian kelompok yang terjebak (> 1.1 x "
          f"{terbaik:.2f})")
    print("  n_init     forgy   pp_asli  pp_serakah")
    for m in (1, 2, 5, 10, 20, 50):
        baris = []
        for nama in ("forgy", "pp_asli", "pp_serakah"):
            J = hasil[nama][: 1000 // m * m].reshape(-1, m).min(axis=1)
            baris.append((J > 1.1 * terbaik).mean())
        print(f"  {m:6d}  " + "  ".join(f"{b:8.3f}" for b in baris))
