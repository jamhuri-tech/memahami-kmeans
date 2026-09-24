"""Bab 11: K-means pada enam data berlabel.

ARI terhadap label sebenarnya, dan inersia dua partisi: partisi
K-means dan partisi menurut label sebenarnya.
"""
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from bab04_data import BENIH
from bab04_inersia import inersia
from bab11_data import enam_data

if __name__ == "__main__":
    print("data        K    ARI   J K-means   J label benar")
    for nama, X, y in enam_data():
        K = y.max() + 1
        km = KMeans(K, n_init=10, random_state=BENIH).fit(X)
        print(f"{nama:9s} {K:3d}  {adjusted_rand_score(y, km.labels_):5.3f}"
              f"  {km.inertia_:10.1f}  {inersia(X, y, K):14.1f}")
