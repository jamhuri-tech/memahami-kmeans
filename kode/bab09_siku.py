"""Bab 9: kurva inersia J(K) untuk K = 1..10 pada keempat data.

Kolom berisi J(K) / J(1), yaitu bagian jumlah kuadrat total yang
belum dijelaskan partisi (1 - B/T).
"""
import numpy as np

from bab09_data import kmeans, semua_data

if __name__ == "__main__":
    data = semua_data()
    J = {nama: [kmeans(X, K).inertia_ for K in range(1, 11)]
         for nama, X, _ in data}
    print(" K " + "".join(f"{nama:>10s}" for nama, _, _ in data))
    for K in range(1, 11):
        print(f"{K:2d} " + "".join(f"{J[nama][K - 1] / J[nama][0]:10.4f}"
                                   for nama, _, _ in data))
    Js = np.array(J["seragam"])
    print("seragam, K x J(K) / J(1), K = 2..10:")
    print("  " + " ".join(f"{k * j / Js[0]:.2f}"
                          for k, j in zip(range(2, 11), Js[1:])))
