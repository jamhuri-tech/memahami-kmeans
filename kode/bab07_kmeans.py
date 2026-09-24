"""Bab 7: kelas KMeansKita, tiruan KMeans(algorithm="lloyd").

Dicetak di naskah sebagai Listing 7.1 (fit), 7.2 (satu jalan Lloyd),
dan 7.3 (klaster kosong dan kesamaan partisi). Inisialisasi yang
tersedia di bab ini: "random" dan larik centroid. k-means++ ditambahkan
di Bab 8 lewat argumen init berupa fungsi.
"""
import numpy as np


class KMeansKita:
    def __init__(self, K, init="random", n_init="auto",
                 max_iter=300, tol=1e-4, random_state=None):
        self.K, self.init, self.n_init = K, init, n_init
        self.max_iter, self.tol = max_iter, tol
        self.random_state = random_state

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        w = np.ones(n) if sample_weight is None \
            else np.asarray(sample_weight, dtype=float)
        # toleransi relatif terhadap rata-rata variansi peubah
        tol = self.tol * np.var(X, axis=0).mean()
        rng = np.random.RandomState(self.random_state)
        larik = not isinstance(self.init, str) and \
            not callable(self.init)
        if larik or self.n_init != "auto":
            ulang = 1 if larik else self.n_init
        else:
            ulang = 10 if self.init == "random" else 1
        rata = X.mean(axis=0)
        Xc = X - rata                          # pemusatan
        terbaik = None
        for _ in range(ulang):
            if larik:
                C0 = np.array(self.init, dtype=float) - rata
            elif callable(self.init):
                C0 = self.init(Xc, self.K, rng, w)
            else:
                C0 = Xc[rng.choice(n, self.K, replace=False,
                                   p=w / w.sum())]
            hasil = lloyd_satu(Xc, w, C0, self.max_iter, tol)
            label, J = hasil[0], hasil[1]
            if terbaik is None:
                terbaik = hasil
            elif J < terbaik[1]:
                if not sama_partisi(label, terbaik[0]):
                    terbaik = hasil
        label, J, C, n_iter = terbaik
        self.labels_, self.inertia_ = label, J
        self.cluster_centers_ = C + rata
        self.n_iter_ = n_iter
        return self

    def predict(self, X):
        C = self.cluster_centers_
        D = (C ** 2).sum(axis=1) - 2 * np.asarray(X) @ C.T
        return D.argmin(axis=1)


def lloyd_satu(Xc, w, C, max_iter, tol):
    K = len(C)
    label = np.full(len(Xc), -1)
    ketat = False
    for i in range(max_iter):
        lama = label
        # kuadrat jarak tanpa suku ||x||^2
        D = (C ** 2).sum(axis=1) - 2 * Xc @ C.T
        label = D.argmin(axis=1)
        bobot = np.bincount(label, weights=w, minlength=K)
        jumlah = np.zeros_like(C)
        np.add.at(jumlah, label, Xc * w[:, None])
        if (bobot == 0).any():
            pindahkan_kosong(Xc, w, C, label, bobot, jumlah)
        C_baru = rata_rata(jumlah, bobot)
        geser = ((C_baru - C) ** 2).sum()
        C = C_baru
        if np.array_equal(label, lama):
            ketat = True
            break
        if geser <= tol:
            break
    if not ketat:                          # penugasan akhir
        D = (C ** 2).sum(axis=1) - 2 * Xc @ C.T
        label = D.argmin(axis=1)
    J = (w * ((Xc - C[label]) ** 2).sum(axis=1)).sum()
    return label, J, C, i + 1


def pindahkan_kosong(Xc, w, C_lama, label, bobot, jumlah):
    kosong = np.flatnonzero(bobot == 0)
    m = len(kosong)
    jarak = ((Xc - C_lama[label]) ** 2).sum(axis=1)
    if jarak.max() == 0:
        return
    jauh = np.argpartition(jarak, -m)[:-m - 1:-1]
    for k, j in zip(kosong, jauh):
        a = label[j]                   # label j TIDAK diubah
        jumlah[a] -= Xc[j] * w[j]
        jumlah[k] = Xc[j] * w[j]
        bobot[k] = w[j]
        bobot[a] -= w[j]


def rata_rata(jumlah, bobot):
    C = jumlah.copy()
    ada = bobot > 0
    C[ada] /= bobot[ada, None]
    C[~ada] = C[bobot.argmax()]    # ke klaster terbesar
    return C


def sama_partisi(a, b):
    """Sama sampai pertukaran nama klaster?"""
    peta = {}
    for x, y in zip(a, b):
        if peta.setdefault(x, y) != y:
            return False
    return True
