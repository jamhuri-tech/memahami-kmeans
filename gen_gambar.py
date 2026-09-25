# -*- coding: utf-8 -*-
"""Membangkitkan seluruh gambar Matplotlib ke gbr/ dalam dua bentuk:
PDF vektor untuk cetak dan PNG 300 dpi untuk EPUB.

Satu fungsi per gambar, dinamai babNN_nama(), yang memanggil
simpan(fig, "babNN-nama"). Fungsi bernama babNN_* dijalankan otomatis.
Benih acak selalu tetap, supaya gambar tidak berubah setiap build.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENIH = 20260924  # sama dengan seluruh kode/bab*.py
GBR = Path("gbr")

# Sebagian gambar memakai kelas yang sudah ditulis di kode/, supaya
# logikanya tidak terduplikasi di dua tempat.
sys.path.insert(0, str(Path(__file__).parent / "kode"))

# Warna mengikuti preamble.tex.
BIRU = "#1B3B6F"
HIJAU = "#1E6F5C"
JINGGA = "#B85C00"
MERAH = "#9B1B30"
ABU = "#5A6472"
ABU_GARIS = "#C9CED6"
BIRU_MUDA = "#E8EEF7"
HIJAU_MUDA = "#E6F2EF"
JINGGA_MUDA = "#FDF0E3"
MERAH_MUDA = "#FBE9EC"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": ABU_GARIS,
    "axes.labelcolor": ABU,
    "axes.titlesize": 8,
    "axes.titlecolor": BIRU,
    "xtick.color": ABU,
    "ytick.color": ABU,
    "text.color": ABU,
    "grid.color": ABU_GARIS,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def angka(v, n=3):
    """Angka dengan koma desimal, sesuai kaidah bahasa Indonesia."""
    return f"{v:.{n}f}".replace(".", ",")


def angka_mat(v, n=3):
    """Seperti angka(), untuk mode matematika: koma tanpa spasi."""
    return angka(v, n).replace(",", "{,}")


def _koma(fig):
    """Mengubah pemisah desimal pada label sumbu menjadi koma.

    Sumbu berskala logaritmik dilewati, karena labelnya berupa pangkat
    sepuluh dan tidak memuat pemisah desimal.
    """
    from matplotlib.ticker import FuncFormatter
    rapi = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        kunci = getattr(ax, "_label_terkunci", set())
        if "x" not in kunci and ax.get_xscale() == "linear":
            ax.xaxis.set_major_formatter(rapi)
        if "y" not in kunci and ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(rapi)


def kunci_label(ax, *sumbu):
    """Menandai sumbu yang labelnya kita tetapkan sendiri."""
    ax._label_terkunci = getattr(ax, "_label_terkunci",
                                 set()) | set(sumbu)


def simpan(fig, nama):
    _koma(fig)
    GBR.mkdir(exist_ok=True)
    fig.savefig(GBR / f"{nama}.pdf", bbox_inches="tight")
    fig.savefig(GBR / f"{nama}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("gbr/" + nama)


def _rapikan(ax):
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)


# Fungsi gambar ditambahkan bab demi bab: babNN_nama().

WARNA_K = [BIRU, JINGGA, HIJAU, MERAH, ABU]


def _warna(label):
    return [WARNA_K[k % len(WARNA_K)] for k in label]


# =====================================================================
#  Bab 4 -- Masalah K-means: inersia dan partisi
# =====================================================================
def _gambar_partisi(ax, X, label, K, judul):
    from bab04_inersia import centroid
    from bab04_data import NAMA
    mu = centroid(X, label, K)
    for i, (x, y) in enumerate(X):
        k = label[i]
        ax.plot([x, mu[k, 0]], [y, mu[k, 1]], color=WARNA_K[k], lw=0.5,
                alpha=0.6, zorder=1)
        ax.scatter(x, y, s=14, color=WARNA_K[k], zorder=3, lw=0)
        dekat = np.hypot(*(mu[k] - (x, y))) < 0.6
        ax.annotate(NAMA[i], (x, y), xytext=(-8 if dekat else 3, 2),
                    textcoords="offset points", fontsize=6, color=ABU)
    ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=30, lw=1.2,
               color=[WARNA_K[k] for k in range(K)], zorder=4)
    ax.set_title(judul, fontsize=7)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.7, 5.7)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 10, 2))
    ax.set_yticks(range(0, 6, 1))
    ax.grid(alpha=0.3, lw=0.4)
    _rapikan(ax)


def bab04_partisi():
    from bab04_data import delapan_titik
    from bab04_inersia import inersia
    X = delapan_titik()
    P = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    Q = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.0))
    _gambar_partisi(ax[0], X, P, 2,
                    rf"$P$: $J = {angka_mat(inersia(X, P, 2), 2)}$")
    _gambar_partisi(ax[1], X, Q, 2,
                    rf"$Q$: $J = {angka_mat(inersia(X, Q, 2), 2)}$")
    fig.tight_layout(w_pad=1.0)
    simpan(fig, "bab04-partisi")


def bab04_voronoi():
    from sklearn.cluster import KMeans
    from bab04_data import gumpalan
    X, _ = gumpalan()
    km = KMeans(5, n_init=10, random_state=BENIH).fit(X)
    mu = km.cluster_centers_
    urut = np.argsort(mu[:, 0] + 0.01 * mu[:, 1])
    mu = mu[urut]
    gx, gy = np.meshgrid(np.linspace(-3.5, 12.5, 500),
                         np.linspace(-3.5, 9.5, 400))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    sel = ((G[:, None, :] - mu[None]) ** 2).sum(2).argmin(1)
    lab = ((X[:, None, :] - mu[None]) ** 2).sum(2).argmin(1)
    from matplotlib.colors import ListedColormap
    muda = ListedColormap([BIRU_MUDA, JINGGA_MUDA, HIJAU_MUDA,
                           MERAH_MUDA, "#EEF0F3"])
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    ax.imshow(sel.reshape(gx.shape), origin="lower", cmap=muda,
              extent=(-3.5, 12.5, -3.5, 9.5), interpolation="nearest",
              aspect="auto", vmin=-0.5, vmax=4.5)
    ax.scatter(X[:, 0], X[:, 1], s=3, c=_warna(lab), lw=0)
    ax.scatter(mu[:, 0], mu[:, 1], marker="x", s=40, lw=1.4,
               color="black")
    # batas sel: garis bagi tegak lurus, digambar lewat kontur label
    ax.contour(gx, gy, sel.reshape(gx.shape), levels=np.arange(0.5, 4),
               colors=ABU, linewidths=0.5)
    ax.set_xlim(-3.5, 12.5)
    ax.set_ylim(-3.5, 9.5)
    ax.set_aspect("equal")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-voronoi")


def bab04_sebaran():
    from bab04_data import delapan_titik
    from bab04_inersia import inersia
    from bab04_semua import partisi, titik_tetap
    X = delapan_titik()
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.8), sharey=False)
    for s, K in zip(ax, (2, 3)):
        J, tetap = [], []
        for p in partisi(len(X), K):
            j = inersia(X, p, K)
            J.append(j)
            if titik_tetap(X, p, K):
                tetap.append(j)
        s.hist(J, bins=40, color=ABU_GARIS, lw=0)
        for j in tetap:
            s.axvline(j, color=JINGGA, lw=0.8)
        s.axvline(min(J), color=BIRU, lw=1.2)
        s.set_title(f"$K = {K}$: {len(J)} partisi, "
                    f"{len(tetap)} titik tetap", fontsize=7)
        s.set_xlabel("$J$")
        _rapikan(s)
    ax[0].set_ylabel("banyaknya partisi")
    fig.tight_layout(w_pad=1.0)
    simpan(fig, "bab04-sebaran")


def bab04_satudimensi():
    from sklearn.cluster import KMeans
    from bab04_data import garis
    from bab04_satudimensi import kmeans_1d_tepat
    x = garis()
    J, batas = kmeans_1d_tepat(x, 4)
    lab_tepat = np.searchsorted(np.array(batas), np.arange(len(x)),
                                side="right")
    # awal acak yang paling sering berakhir di J sekitar 62,08
    for s in range(1000):
        km = KMeans(4, init="random", n_init=1,
                    random_state=s).fit(x[:, None])
        if abs(km.inertia_ - 62.0758) < 1e-3:
            break
    urut = np.argsort(km.cluster_centers_.ravel())
    lab_lokal = np.argsort(urut)[km.labels_]
    fig, ax = plt.subplots(figsize=(4.7, 1.3))
    for baris, lab, nama, j in ((1, lab_tepat, "optimum", J),
                                (0, lab_lokal, "minimum lokal",
                                 km.inertia_)):
        ax.scatter(x, np.full(len(x), baris), s=8, c=_warna(lab), lw=0)
        for k in range(4):
            ax.plot(x[lab == k].mean(), baris + 0.28, marker="v",
                    ms=4, color=WARNA_K[k])
        ax.text(-2.6, baris, f"{nama}\n$J = {angka_mat(j, 2)}$",
                fontsize=6, va="center", ha="right")
    ax.set_ylim(-0.5, 1.6)
    ax.set_yticks([])
    ax.set_xlim(-2.4, 12)
    ax.spines["left"].set_visible(False)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab04-satudimensi")



# =====================================================================
#  Bab 5 -- Algoritma Lloyd langkah demi langkah
# =====================================================================
def _garis_sumbu(ax, a, b, **kw):
    """Garis sumbu ruas ab (batas dua sel Voronoi) di jendela ax."""
    m = (a + b) / 2
    arah = np.array([-(b - a)[1], (b - a)[0]])
    arah = arah / np.hypot(*arah)
    s = np.linspace(-20, 20, 2)
    ax.plot(m[0] + s * arah[0], m[1] + s * arah[1], **kw)


def bab05_tangan():
    from bab04_data import NAMA, delapan_titik
    from bab05_lloyd import lloyd
    X = delapan_titik()
    C = X[[0, 1]].copy()
    _, _, riwayat = lloyd(X, C)
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.75), sharey=True)
    lama = C
    for t, s in enumerate(ax):
        _, J_baru, baru, label = riwayat[t]
        _garis_sumbu(s, lama[0], lama[1], color=ABU, lw=0.6, ls="--")
        s.scatter(X[:, 0], X[:, 1], s=10, c=_warna(label), lw=0, zorder=3)
        for k in range(2):
            s.scatter(*lama[k], marker="x", s=18, lw=0.9,
                      color=WARNA_K[k], alpha=0.45, zorder=4)
            s.annotate("", xy=baru[k], xytext=lama[k],
                       arrowprops=dict(arrowstyle="->", lw=0.7,
                                       color=WARNA_K[k]))
            s.scatter(*baru[k], marker="x", s=24, lw=1.2,
                      color=WARNA_K[k], zorder=4)
        s.set_title(f"iterasi {t + 1}: $J = {angka_mat(J_baru, 2)}$",
                    fontsize=6.5)
        s.set_xlim(0, 9)
        s.set_ylim(-0.7, 5.7)
        s.set_aspect("equal")
        s.set_xticks([0, 4, 8])
        _rapikan(s)
        lama = baru
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab05-tangan")


def bab05_inersia():
    from bab04_data import BENIH, delapan_titik, gumpalan
    from bab05_lloyd import lloyd
    from bab05_monoton import barisan_J
    X = delapan_titik()
    _, _, r8 = lloyd(X, X[[0, 1]].copy())
    Xg, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    C0 = Xg[rng.choice(len(Xg), 5, replace=False)]
    _, _, rg = lloyd(Xg, C0)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.8))
    for s, r, judul in ((ax[0], r8, "delapan titik"),
                        (ax[1], rg, "gumpalan, satu awal acak")):
        J = barisan_J(r)
        tt = np.arange(len(J)) / 2 + 0.5
        s.step(tt, J, where="post", color=BIRU, lw=0.9)
        s.plot(tt[0::2], J[0::2], "o", ms=2.5, color=JINGGA,
               label="sesudah penugasan")
        s.plot(tt[1::2], J[1::2], "s", ms=2.5, color=HIJAU,
               label="sesudah pembaruan")
        s.set_title(judul, fontsize=7)
        s.set_xlabel("iterasi")
        _rapikan(s)
    ax[0].set_ylabel("$J$")
    ax[0].legend(fontsize=5.5)
    fig.tight_layout(w_pad=1.0)
    simpan(fig, "bab05-inersia")


def bab05_lintasan():
    from bab04_data import BENIH, gumpalan
    from bab05_lloyd import lloyd
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    contoh = {}
    for _ in range(1000):
        C0 = X[rng.choice(len(X), 5, replace=False)]
        C, label, r = lloyd(X, C0)
        J = ((X - C[label]) ** 2).sum()
        kunci = "baik" if J < 909 else ("buruk" if J > 1500 else None)
        if kunci and kunci not in contoh:
            contoh[kunci] = (C0, r, J)
        if len(contoh) == 2:
            break
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2), sharey=True)
    for s, kunci in zip(ax, ("baik", "buruk")):
        C0, r, J = contoh[kunci]
        label = r[-1][3]
        s.scatter(X[:, 0], X[:, 1], s=1.5, c=_warna(label), lw=0,
                  alpha=0.5)
        jalur = np.array([C0] + [c for _, _, c, _ in r])
        for k in range(5):
            s.plot(jalur[:, k, 0], jalur[:, k, 1], "-", color="black",
                   lw=0.6)
            s.plot(*jalur[0, k], "o", ms=3, mfc="white", mec="black",
                   mew=0.6)
            s.plot(*jalur[-1, k], "x", ms=4, color="black", mew=1.1)
        s.set_title(f"{len(r)} iterasi, $J = {angka_mat(J, 1)}$",
                    fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.6)
    simpan(fig, "bab05-lintasan")


def bab05_lokal():
    from bab04_data import BENIH, gumpalan
    from bab05_lokal import inersia_akhir
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    J = np.array([inersia_akhir(X, X[rng.choice(len(X), 5,
                                                 replace=False)])
                  for _ in range(1000)])
    fig, ax = plt.subplots(figsize=(4.7, 1.7))
    ax.hist(J, bins=np.linspace(880, 2280, 71), color=ABU_GARIS, lw=0)
    ax.hist(J[J < 1.1 * J.min()], bins=np.linspace(880, 2280, 71),
            color=BIRU, lw=0)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1500)
    ax.set_xlabel("$J$ akhir")
    ax.set_ylabel("banyaknya awal")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab05-lokal")


# =====================================================================
#  Bab 6 -- K-means sebagai optimisasi
# =====================================================================
def bab06_lanskap():
    from bab05_lloyd import lloyd
    from bab06_fungsi import tiga_kelompok
    X = tiga_kelompok()
    x = X.ravel()
    g = np.linspace(-2, 11, 261)
    M1, M2 = np.meshgrid(g, g)
    Fg = np.minimum((x - M1[..., None]) ** 2,
                    (x - M2[..., None]) ** 2).sum(-1)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    lv = np.quantile(Fg, np.linspace(0.005, 0.9, 24))
    ax.contour(M1, M2, Fg, levels=np.unique(lv), colors=ABU_GARIS,
               linewidths=0.5)
    ax.plot([-2, 11], [-2, 11], color=ABU, lw=0.4, ls=":")
    for (a, b), w in zip(((-1.5, 3.0), (2.5, 3.5), (6.0, 10.5),
                          (-1.0, 10.0)), (BIRU, JINGGA, HIJAU, MERAH)):
        C0 = np.array([[a], [b]])
        _, _, r = lloyd(X, C0)
        jalur = np.array([C0.ravel()] + [c.ravel() for _, _, c, _ in r])
        ax.plot(jalur[:, 0], jalur[:, 1], "-o", ms=2.2, lw=0.8, color=w)
        ax.plot(*jalur[-1], "x", ms=6, mew=1.4, color=w)
    ax.plot(x, np.full(len(x), -1.8), "|", ms=3, color=ABU)
    ax.plot(np.full(len(x), -1.8), x, "_", ms=3, color=ABU)
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.set_xlim(-2, 11)
    ax.set_ylim(-2, 11)
    ax.set_aspect("equal")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-lanskap")


def bab06_mm():
    from bab05_lloyd import jarak2
    from bab06_fungsi import tiga_kelompok
    X = tiga_kelompok()
    x = X.ravel()
    mu1 = 0.5
    g = np.linspace(-1, 12, 600)
    Fs = np.array([np.minimum((x - mu1) ** 2, (x - m) ** 2).sum()
                   for m in g])
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    ax.plot(g, Fs, color=BIRU, lw=1.1, label=r"$F(\mu_1, \mu_2)$")
    m = 3.0
    for t, w in zip(range(3), (JINGGA, HIJAU, MERAH)):
        dua = (x - m) ** 2 < (x - mu1) ** 2
        tetap1 = ((x[~dua] - mu1) ** 2).sum()
        G = tetap1 + ((x[dua][:, None] - g[None, :]) ** 2).sum(0)
        ax.plot(g, G, color=w, lw=0.8, ls="--",
                label=rf"$G(\cdot \mid \mu_2^{{({t})}})$")
        Fm = np.minimum((x - mu1) ** 2, (x - m) ** 2).sum()
        ax.plot(m, Fm, "o", ms=3, color=w)
        m = x[dua].mean()
    ax.plot(m, np.minimum((x - mu1) ** 2, (x - m) ** 2).sum(), "o",
            ms=3, color=BIRU)
    ax.set_ylim(150, 950)
    ax.set_xlim(1, 12)
    ax.set_xlabel(r"$\mu_2$")
    ax.legend(fontsize=5.5, ncol=2, loc="upper center")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-mm")


def bab06_gd():
    from bab04_data import BENIH, gumpalan
    from bab05_lloyd import lloyd
    from bab06_fungsi import F, gradien_hessian
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    C0 = X[rng.choice(len(X), 5, replace=False)]
    C_akhir, _, r = lloyd(X, C0)
    J_akhir = F(X, C_akhir)
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    Jl = [F(X, C0)] + [F(X, c) for _, _, c, _ in r]
    ax.semilogy(np.arange(len(Jl)), np.maximum(np.array(Jl) - J_akhir,
                                               1e-9),
                "-o", ms=2.5, color=BIRU, lw=1.1, label="Lloyd (Newton)")
    _, h = gradien_hessian(X, C0)
    nmaks = h.max() / 2
    for c, w in ((0.1, ABU), (0.5, HIJAU), (1.5, JINGGA), (1.9, MERAH)):
        C = C0.copy()
        J = [F(X, C)]
        for _ in range(60):
            g, _ = gradien_hessian(X, C)
            C = C - c / (2 * nmaks) * g
            J.append(F(X, C))
        ax.semilogy(np.maximum(np.array(J) - J_akhir, 1e-9), color=w,
                    lw=0.8, label=f"GD, $c = {angka_mat(c, 1)}$")
    ax.set_xlim(0, 60)
    ax.set_ylim(1e-9, 1e4)
    ax.set_xlabel("langkah")
    ax.set_ylabel(r"$F - F_{\mathrm{akhir}}$")
    ax.legend(fontsize=5.5, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-gd")


def bab06_hartigan():
    from bab04_data import BENIH, gumpalan
    from bab05_lloyd import lloyd
    from bab06_hartigan import hartigan, inersia_label
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    Jl, Jh = [], []
    for _ in range(1000):
        C0 = X[rng.choice(len(X), 5, replace=False)]
        _, label, _ = lloyd(X, C0)
        Jl.append(inersia_label(X, label, 5))
        Jh.append(inersia_label(X, hartigan(X, label, 5)[0], 5))
    Jl, Jh = np.array(Jl), np.array(Jh)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(Jl, Jh, "o", ms=2.5, color=BIRU, alpha=0.35, mew=0)
    ax.plot([850, 2300], [850, 2300], color=ABU, lw=0.5, ls=":")
    ax.set_xlabel("$J$ Lloyd")
    ax.set_ylabel("$J$ Lloyd lalu Hartigan")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab06-hartigan")


# =====================================================================
#  Bab 7 -- K-means dari nol
# =====================================================================
def bab07_presisi():
    from sklearn.cluster import KMeans
    from bab04_data import BENIH, gumpalan
    X0, _ = gumpalan()
    C0 = KMeans(5, n_init=10, random_state=BENIH).fit(X0) \
        .cluster_centers_
    geser = np.logspace(4, 10, 25)
    urai, pusat = [], []
    for c in geser:
        X, C = X0 + c, C0 + c
        benar = ((X[:, None] - C[None]) ** 2).sum(axis=2).argmin(1)
        urai.append((((C ** 2).sum(1) - 2 * X @ C.T).argmin(1)
                     != benar).mean())
        m = X.mean(axis=0)
        pusat.append(((((C - m) ** 2).sum(1)
                       - 2 * (X - m) @ (C - m).T).argmin(1)
                      != benar).mean())
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    ax.semilogx(geser, np.array(urai) * 100, "-o", ms=2.5, color=MERAH,
                lw=0.9, label="terurai, tanpa pemusatan")
    ax.semilogx(geser, np.array(pusat) * 100, "-s", ms=2.5, color=BIRU,
                lw=0.9, label="terurai, sesudah pemusatan")
    ax.axhline(80, color=ABU, lw=0.5, ls=":")
    ax.set_xlabel("pergeseran data $c$")
    ax.set_ylabel("label salah (%)")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-presisi")


def bab07_tol():
    from bab04_data import BENIH, gumpalan
    from bab07_kmeans import KMeansKita
    X, _ = gumpalan(5000, BENIH, 2.0)
    v = np.var(X, axis=0).mean()
    acuan = KMeansKita(5, n_init=1, tol=0, random_state=3).fit(X)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9))
    # jalankan ulang dengan max_iter = 1, 2, ... untuk mencatat
    # pergeseran centroid dan inersia di setiap iterasi
    geser, J = [], []
    C_lama = None
    for m in range(1, acuan.n_iter_ + 1):
        a = KMeansKita(5, n_init=1, tol=0, max_iter=m,
                       random_state=3).fit(X)
        if C_lama is not None:
            geser.append(((a.cluster_centers_ - C_lama) ** 2).sum())
        C_lama = a.cluster_centers_
        J.append(a.inertia_)
    it = np.arange(2, acuan.n_iter_ + 1)
    ax[0].semilogy(it, np.maximum(geser, 1e-12), "-o", ms=2.3,
                   color=BIRU, lw=0.9)
    for tol, w in ((1e-4, JINGGA), (1e-2, MERAH)):
        ax[0].axhline(tol * v, color=w, lw=0.7, ls="--")
        ax[0].text(acuan.n_iter_, tol * v * 1.6,
                   rf"tol $= 10^{{{int(np.log10(tol))}}}$",
                   fontsize=5.5, ha="right", color=w)
    ax[0].set_xlabel("iterasi")
    ax[0].set_ylabel("pergeseran centroid")
    ax[1].semilogy(np.arange(1, len(J) + 1),
                   np.array(J) - J[-1] + 1e-3, "-o", ms=2.3,
                   color=HIJAU, lw=0.9)
    ax[1].set_xlabel("iterasi")
    ax[1].set_ylabel(r"$J - J_{\mathrm{akhir}}$")
    for s in ax:
        _rapikan(s)
    fig.tight_layout(w_pad=1.0)
    simpan(fig, "bab07-tol")


def bab07_waktu():
    import time
    from sklearn.cluster import KMeans
    from bab07_kmeans import KMeansKita
    rng = np.random.default_rng(BENIH)
    ukuran = [1000, 3000, 10000, 30000, 100000, 300000]
    t_kita, t_sk = [], []
    for n in ukuran:
        X = rng.normal(size=(n, 10)) + \
            rng.normal(0, 3, size=(8, 10))[rng.integers(0, 8, n)]
        C0 = X[:8].copy()
        waktu = []
        for kelas in (KMeansKita, KMeans):
            terbaik = np.inf
            for _ in range(3):
                t0 = time.perf_counter()
                kelas(8, init=C0, n_init=1, max_iter=20, tol=0).fit(X)
                terbaik = min(terbaik, time.perf_counter() - t0)
            waktu.append(terbaik)
        t_kita.append(waktu[0])
        t_sk.append(waktu[1])
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    ax.loglog(ukuran, t_kita, "-o", ms=2.5, color=JINGGA, lw=0.9,
              label="KMeansKita (NumPy)")
    ax.loglog(ukuran, t_sk, "-s", ms=2.5, color=BIRU, lw=0.9,
              label="KMeans scikit-learn")
    ax.set_xlabel("banyaknya titik $n$")
    ax.set_ylabel("waktu (detik)")
    ax.legend(fontsize=5.5)
    kunci_label(ax, "x", "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab07-waktu")
    print("  waktu kita:", [f"{v:.3f}" for v in t_kita])
    print("  waktu sklearn:", [f"{v:.3f}" for v in t_sk])


# =====================================================================
#  Bab 8 -- Inisialisasi dan k-means++
# =====================================================================
def bab08_awal():
    from bab08_awal import forgy, kisi, kotak, partisi, pp_serakah
    X = kisi()
    Xc = X - X.mean(axis=0)
    w = np.ones(len(X))
    fig, ax = plt.subplots(2, 2, figsize=(4.4, 4.4), sharex=True,
                           sharey=True)
    for s, (nama, f) in zip(ax.ravel(),
                            (("Forgy", forgy),
                             ("partisi acak", partisi),
                             ("seragam dalam kotak", kotak),
                             ("k-means++ serakah", pp_serakah))):
        C = f(Xc, 25, np.random.RandomState(BENIH), w) + X.mean(axis=0)
        s.scatter(X[:, 0], X[:, 1], s=1, color=ABU_GARIS, lw=0)
        s.scatter(C[:, 0], C[:, 1], marker="x", s=14, lw=1.0,
                  color=MERAH)
        s.set_title(nama, fontsize=7)
        s.set_aspect("equal")
        s.set_xticks([0, 8, 16])
        s.set_yticks([0, 8, 16])
        _rapikan(s)
    fig.tight_layout(w_pad=0.5, h_pad=0.8)
    simpan(fig, "bab08-awal")


def bab08_d2():
    from bab04_data import gumpalan
    from bab08_plusplus import jarak2_ke
    X, _ = gumpalan()
    xx = (X ** 2).sum(1)
    rng = np.random.RandomState(4)
    idx = [rng.choice(len(X))]
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.1), sharey=True)
    for s in ax:
        d2 = jarak2_ke(X, xx, idx).min(axis=0)
        p = d2 / d2.sum()
        urut = np.argsort(p)
        sc = s.scatter(X[urut, 0], X[urut, 1], c=p[urut] * 100, s=3,
                       cmap="Blues", lw=0, vmin=0)
        s.scatter(X[idx, 0], X[idx, 1], marker="x", s=30, lw=1.3,
                  color=MERAH)
        s.set_title(f"{len(idx)} centroid terpilih", fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
        idx.append(int(np.argmax(rng.uniform(size=1)[0]
                                 < np.cumsum(p))))
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("peluang terpilih (%)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    simpan(fig, "bab08-d2")


def bab08_sebaran():
    from bab08_awal import CARA, jalankan, kisi
    X = kisi()
    hasil = {nama: jalankan(X, 25, f)[1] for nama, f in CARA}
    terbaik = min(j.min() for j in hasil.values())
    nama_tampil = {"forgy": "Forgy", "partisi": "partisi acak",
                   "kotak": "seragam dalam kotak",
                   "pp_asli": "k-means++ asli",
                   "pp_serakah": "k-means++ serakah"}
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    for (nama, J), w in zip(hasil.items(), (ABU, HIJAU, JINGGA,
                                           BIRU_MUDA, BIRU)):
        r = np.sort(J / terbaik)
        ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post",
                color=w if nama != "pp_asli" else "#6E8FC7", lw=1.0,
                label=nama_tampil[nama])
    ax.axvline(1.1, color=ABU, lw=0.5, ls=":")
    ax.set_xlim(0.98, 3.0)
    ax.set_xlabel(r"$J$ akhir / $J$ terbaik")
    ax.set_ylabel("bagian awal")
    ax.legend(fontsize=5.5, loc="lower right")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-sebaran")


def bab08_ninit():
    from bab08_awal import CARA, jalankan, kisi
    X = kisi()
    hasil = {nama: jalankan(X, 25, f)[1] for nama, f in CARA
             if nama in ("forgy", "pp_asli", "pp_serakah")}
    terbaik = min(j.min() for j in hasil.values())
    m_all = np.array([1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 40, 50])
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for (nama, J), w, lab in zip(hasil.items(), (ABU, "#6E8FC7", BIRU),
                                 ("Forgy", "k-means++ asli",
                                  "k-means++ serakah")):
        frac = [(J[: 1000 // m * m].reshape(-1, m).min(axis=1)
                 > 1.1 * terbaik).mean() for m in m_all]
        ax.plot(m_all, frac, "-o", ms=2.5, color=w, lw=0.9, label=lab)
        p1 = (J > 1.1 * terbaik).mean()
        ax.plot(m_all, p1 ** m_all, ":", color=w, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    kunci_label(ax, "x")
    ax.set_xlabel("n_init")
    ax.set_ylabel("bagian terjebak")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab08-ninit")


# =====================================================================
#  Bab 9 -- Memilih banyaknya klaster
# =====================================================================
def bab09_siku():
    from bab09_data import kmeans, semua_data
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    for (nama, X, _), w in zip(semua_data(), (BIRU, JINGGA, MERAH, HIJAU)):
        J = np.array([kmeans(X, K).inertia_ for K in range(1, 11)])
        ax.plot(range(1, 11), J / J[0], "-o", ms=2.5, lw=0.9, color=w,
                label=nama)
    ax.set_xlabel("$K$")
    ax.set_ylabel(r"$J(K) / J(1)$")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab09-siku")


def bab09_seragam():
    from bab09_data import kmeans, semua_data
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.75), sharey=True)
    for s, (nama, X, _) in zip(ax, semua_data()[:3]):
        km = kmeans(X, 5 if nama != "seragam" else 4)
        s.scatter(X[:, 0], X[:, 1], s=1.2, c=_warna(km.labels_), lw=0)
        C = km.cluster_centers_
        s.scatter(C[:, 0], C[:, 1], marker="x", s=16, lw=1.0,
                  color="black")
        s.set_title(f"{nama}, $K = {len(C)}$", fontsize=6.5)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab09-seragam")


def bab09_silhouette():
    from bab09_data import kmeans, semua_data
    from bab09_ukuran import silhouette
    X = semua_data()[0][1]
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 2.2), sharey=True)
    for s, K in zip(ax, (4, 5, 6)):
        lab = kmeans(X, K).labels_
        sil = silhouette(X, lab)
        y = 0
        for k in range(K):
            v = np.sort(sil[lab == k])
            s.barh(np.arange(y, y + len(v)), v, height=1.0,
                   color=WARNA_K[k % 5], lw=0)
            y += len(v) + 8
        s.axvline(sil.mean(), color="black", lw=0.6, ls="--")
        s.set_title(f"$K = {K}$, rata-rata {angka(sil.mean(), 2)}",
                    fontsize=6.5)
        s.set_yticks([])
        s.set_xlim(-0.2, 1)
        s.set_xlabel("$s_i$")
        _rapikan(s)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab09-silhouette")


def bab09_gap():
    from bab09_data import semua_data
    from bab09_gap import gap
    fig, ax = plt.subplots(1, 4, figsize=(4.7, 1.6), sharex=True)
    for s, (nama, X, _), w in zip(ax, semua_data(),
                                   (BIRU, JINGGA, MERAH, HIJAU)):
        G, sd, Kp = gap(X, 10)
        s.errorbar(range(1, 11), G, yerr=sd, fmt="-o", ms=2, lw=0.8,
                   color=w, elinewidth=0.6, capsize=1.2)
        if Kp:
            s.plot(Kp, G[Kp - 1], "o", ms=5, mfc="none", mec="black",
                   mew=0.8)
        s.set_title(f"{nama}: {Kp if Kp else '≥ 10'}", fontsize=6.5)
        s.set_xticks([1, 5, 10])
        s.set_xlabel("$K$")
        _rapikan(s)
    ax[0].set_ylabel("Gap")
    fig.tight_layout(w_pad=0.3)
    simpan(fig, "bab09-gap")


# =====================================================================
#  Bab 10 -- Menilai hasil clustering
# =====================================================================
def _digits_k10():
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from bab10_ukuran import tabel
    X, y = load_digits(return_X_y=True)
    km = KMeans(10, n_init=10, random_state=BENIH).fit(X)
    T = tabel(y, km.labels_)
    _, kolom = linear_sum_assignment(-T)
    return X, y, km, T[:, kolom], kolom


def bab10_tabel():
    X, y, km, T, kolom = _digits_k10()
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.imshow(T, cmap="Blues", vmin=0, vmax=T.max())
    for i in range(10):
        for j in range(10):
            if T[i, j]:
                ax.text(j, i, str(T[i, j]), ha="center", va="center",
                        fontsize=5, color="white" if T[i, j] > 90
                        else ABU)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("klaster (diurutkan menurut padanan)")
    ax.set_ylabel("angka sebenarnya")
    kunci_label(ax, "x", "y")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    simpan(fig, "bab10-tabel")


def bab10_centroid():
    X, y, km, T, kolom = _digits_k10()
    fig, ax = plt.subplots(1, 10, figsize=(4.7, 0.75))
    for j, s in enumerate(ax):
        k = kolom[j]
        s.imshow(km.cluster_centers_[k].reshape(8, 8), cmap="gray_r",
                 vmin=0, vmax=16)
        s.set_title(f"{j}: {T[j, j]}/{T[:, j].sum()}", fontsize=5)
        s.axis("off")
    fig.tight_layout(w_pad=0.2)
    simpan(fig, "bab10-centroid")


def bab10_kebetulan():
    from sklearn.datasets import load_digits
    from bab10_ukuran import akurasi_hungaria, ari, nmi, purity, rand
    _, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(BENIH)
    Ks = [2, 3, 5, 10, 20, 50, 100, 200, 300]
    nilai = {n: [] for n in ("RI", "purity", "NMI", "ARI")}
    for K in Ks:
        labs = [rng.integers(0, K, len(y)) for _ in range(10)]
        for n, f in (("RI", rand), ("purity", purity), ("NMI", nmi),
                     ("ARI", ari)):
            nilai[n].append(np.mean([f(y, l) for l in labs]))
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for (n, v), w in zip(nilai.items(), (ABU, JINGGA, HIJAU, BIRU)):
        ax.plot(Ks, v, "-o", ms=2.5, lw=0.9, color=w, label=n)
    ax.set_xscale("log")
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.minorticks_off()
    kunci_label(ax, "x")
    ax.set_xlabel("banyaknya klaster acak $K$")
    ax.set_ylabel("nilai ukuran")
    ax.legend(fontsize=5.5, ncol=4, loc="upper left")
    ax.set_ylim(-0.05, 1.05)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-kebetulan")


def bab10_k():
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from bab10_ukuran import akurasi_hungaria, ari, nmi, purity
    X, y = load_digits(return_X_y=True)
    Ks = list(range(4, 31, 2))
    v = {n: [] for n in ("purity", "akurasi Hungaria", "ARI", "NMI")}
    for K in Ks:
        lab = KMeans(K, n_init=10, random_state=BENIH).fit(X).labels_
        v["purity"].append(purity(y, lab))
        v["akurasi Hungaria"].append(akurasi_hungaria(y, lab))
        v["ARI"].append(ari(y, lab))
        v["NMI"].append(nmi(y, lab))
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for (n, val), w in zip(v.items(), (JINGGA, MERAH, BIRU, HIJAU)):
        ax.plot(Ks, val, "-o", ms=2.3, lw=0.9, color=w, label=n)
    ax.axvline(10, color=ABU, lw=0.5, ls=":")
    ax.set_xlabel("$K$")
    ax.set_ylabel("nilai ukuran")
    ax.legend(fontsize=5.5, ncol=2)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab10-k")


# =====================================================================
#  Bab 11 -- Asumsi tersembunyi dan penskalaan
# =====================================================================
def bab11_gagal():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from bab11_data import enam_data
    fig, ax = plt.subplots(2, 3, figsize=(4.7, 3.2))
    for s, (nama, X, y) in zip(ax.ravel(), enam_data()):
        km = KMeans(y.max() + 1, n_init=10, random_state=BENIH).fit(X)
        s.scatter(X[:, 0], X[:, 1], s=1.2, c=_warna(km.labels_), lw=0)
        s.scatter(*km.cluster_centers_.T, marker="x", s=14, lw=1.0,
                  color="black")
        a = adjusted_rand_score(y, km.labels_)
        s.set_title(f"{nama}, ARI {angka(round(a, 2) + 0.0, 2)}",
                    fontsize=6.5)
        s.set_aspect("equal", adjustable="datalim")
        s.set_xticks([])
        s.set_yticks([])
        for sisi in s.spines.values():
            sisi.set_color(ABU_GARIS)
    fig.tight_layout(w_pad=0.3, h_pad=0.6)
    simpan(fig, "bab11-gagal")


def bab11_timpang():
    from sklearn.cluster import KMeans
    from bab11_data import enam_data
    _, X, y = enam_data()[2]
    km = KMeans(3, n_init=10, random_state=BENIH).fit(X)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.0), sharey=True)
    for s, lab, judul in ((ax[0], y, "label sebenarnya"),
                          (ax[1], km.labels_, "K-means")):
        s.scatter(X[:, 0], X[:, 1], s=2, c=_warna(lab), lw=0)
        s.set_title(judul, fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
    ax[1].scatter(*km.cluster_centers_.T, marker="x", s=20, lw=1.1,
                  color="black")
    fig.tight_layout(w_pad=0.5)
    simpan(fig, "bab11-timpang")


def bab11_wine():
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    X, y = load_wine(return_X_y=True)
    Z = StandardScaler().fit_transform(X)
    P = PCA(2).fit_transform(Z)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.1), sharey=True)
    for s, data, judul in ((ax[0], X, "tanpa penskalaan"),
                           (ax[1], Z, "sesudah pembakuan")):
        lab = KMeans(3, n_init=10, random_state=BENIH).fit(data).labels_
        for kelas, m in zip(range(3), ("o", "s", "^")):
            k = y == kelas
            s.scatter(P[k, 0], P[k, 1], s=6, marker=m,
                      c=_warna(lab[k]), lw=0)
        a = adjusted_rand_score(y, lab)
        s.set_title(f"{judul}, ARI {angka(a, 2)}", fontsize=6.5)
        s.set_xlabel("komponen utama 1")
        _rapikan(s)
    ax[0].set_ylabel("komponen utama 2")
    fig.tight_layout(w_pad=0.5)
    simpan(fig, "bab11-wine")


def bab11_pemutih():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from bab11_data import enam_data
    from bab11_pemutih import putihkan
    _, X, y = enam_data()[1]
    S_total = np.cov(X, rowvar=False)
    S_dalam = sum(np.cov(X[y == k], rowvar=False) for k in range(3)) / 3
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.8))
    for s, Z, judul in ((ax[0], X, "mentah"),
                        (ax[1], putihkan(X, S_total), "kovarians gabungan"),
                        (ax[2], putihkan(X, S_dalam), "kovarians dalam")):
        lab = KMeans(3, n_init=10, random_state=BENIH).fit(Z).labels_
        s.scatter(Z[:, 0], Z[:, 1], s=1.2, c=_warna(lab), lw=0)
        a = adjusted_rand_score(y, lab)
        s.set_title(f"{judul}\nARI {angka(a, 2)}", fontsize=6.3)
        s.set_aspect("equal", adjustable="datalim")
        s.set_xticks([])
        s.set_yticks([])
        for sisi in s.spines.values():
            sisi.set_color(ABU_GARIS)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab11-pemutih")


# =====================================================================
#  Bab 12 -- Dari K-means ke campuran Gaussian
# =====================================================================
def _elips(ax, mu, S, warna, n_sb=2.0):
    from matplotlib.patches import Ellipse
    nilai, vektor = np.linalg.eigh(S)
    sudut = np.degrees(np.arctan2(vektor[1, 1], vektor[0, 1]))
    lebar, tinggi = 2 * n_sb * np.sqrt(nilai[::-1])
    ax.add_patch(Ellipse(mu, lebar, tinggi, angle=sudut, fill=False,
                         color=warna, lw=0.9))


def bab12_batas():
    from matplotlib.colors import to_rgb
    from bab04_data import BENIH, gumpalan
    from bab12_batas import soft_kmeans
    X, _ = gumpalan()
    rng = np.random.default_rng(BENIH)
    C0 = X[rng.choice(len(X), 5, replace=False)]
    warna = np.array([to_rgb(w) for w in WARNA_K])
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.75), sharey=True)
    for s, sigma in zip(ax, (2.0, 1.0, 0.2)):
        C, g, _ = soft_kmeans(X, C0, sigma)
        s.scatter(X[:, 0], X[:, 1], s=1.5, c=g @ warna, lw=0)
        s.scatter(*C.T, marker="x", s=16, lw=1.0, color="black")
        s.set_title(rf"$\sigma = {angka_mat(sigma, 1)}$", fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab12-batas")


def bab12_jenis():
    from sklearn.mixture import GaussianMixture
    from bab11_data import enam_data
    data = enam_data()
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.9))
    for s, (i, jenis) in zip(ax, ((1, "tied"), (2, "spherical"),
                                  (3, "spherical"))):
        nama, X, y = data[i]
        gm = GaussianMixture(y.max() + 1, covariance_type=jenis,
                             n_init=5, random_state=BENIH).fit(X)
        lab = gm.predict(X)
        s.scatter(X[:, 0], X[:, 1], s=1.2, c=_warna(lab), lw=0)
        for k in range(y.max() + 1):
            if jenis == "tied":
                S = gm.covariances_
            else:
                S = np.eye(2) * gm.covariances_[k]
            _elips(s, gm.means_[k], S, "black")
        s.set_title(f"{nama}, {'terikat' if jenis == 'tied' else 'sferis'}",
                    fontsize=6.5)
        s.set_aspect("equal", adjustable="datalim")
        s.set_xticks([])
        s.set_yticks([])
        for sisi in s.spines.values():
            sisi.set_color(ABU_GARIS)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab12-jenis")


def bab12_em():
    from bab11_data import enam_data
    from bab12_em import CampuranKita
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for (nama, X, y), w in zip(enam_data()[:4], (BIRU, JINGGA, HIJAU,
                                                 MERAH)):
        m = CampuranKita(y.max() + 1, tol=1e-8, max_iter=60,
                         random_state=0).fit(X)
        r = np.array(m.riwayat)
        ax.plot(np.arange(1, len(r) + 1), r - r[0], "-", lw=0.9,
                color=w, label=nama)
    ax.set_xlabel("iterasi EM")
    ax.set_ylabel("kenaikan rata-rata log $p(x)$")
    ax.legend(fontsize=5.5)
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-em")


def bab12_ragu():
    from sklearn.mixture import GaussianMixture
    from bab11_data import enam_data
    _, X, y = enam_data()[3]
    gm = GaussianMixture(3, covariance_type="spherical", n_init=5,
                         random_state=BENIH).fit(X)
    p = gm.predict_proba(X).max(axis=1)
    fig, ax = plt.subplots(figsize=(4.0, 2.0))
    sc = ax.scatter(X[:, 0], X[:, 1], s=3, c=p, cmap="viridis_r",
                    vmin=0.34, vmax=1, lw=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("tanggung jawab terbesar", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_aspect("equal")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab12-ragu")


# =====================================================================
#  Bab 13 -- Pencilan dan jarak lain
# =====================================================================
def bab13_pusat():
    rng = np.random.default_rng(BENIH)
    x = np.sort(np.concatenate([rng.normal(2, 0.6, 9), [12.0]]))
    m = np.linspace(-1, 13, 600)
    kuadrat = ((x[:, None] - m) ** 2).sum(0)
    mutlak = np.abs(x[:, None] - m).sum(0)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.8))
    for s, f, nama, pusat, w in ((ax[0], kuadrat, "jumlah kuadrat",
                                  x.mean(), BIRU),
                                 (ax[1], mutlak, "jumlah jarak mutlak",
                                  np.median(x), HIJAU)):
        s.plot(m, f, color=w, lw=1.0)
        s.axvline(pusat, color=w, lw=0.6, ls="--")
        s.plot(x, np.full(len(x), f.min() * 0.0), "|", ms=6,
               color=ABU)
        s.set_title(nama, fontsize=7)
        s.set_xlabel("$m$")
        _rapikan(s)
    fig.tight_layout(w_pad=0.8)
    simpan(fig, "bab13-pusat")


def bab13_pencilan():
    from sklearn.cluster import kmeans_plusplus
    from bab05_lloyd import lloyd
    from bab13_kokoh import kmedoids, trimmed_kmeans
    from bab13_pencilan import dengan_pencilan
    X, y = dengan_pencilan(20)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(axis=2))
    hasil = {}
    for s in range(10):
        C0, idx = kmeans_plusplus(X, 5, random_state=s)
        C, lab, _ = lloyd(X, C0)
        J = ((X - C[lab]) ** 2).sum()
        if "K-means" not in hasil or J < hasil["K-means"][2]:
            hasil["K-means"] = (lab, C, J, None)
        lab, med, J = kmedoids(D, idx)
        if "K-medoids" not in hasil or J < hasil["K-medoids"][2]:
            hasil["K-medoids"] = (lab, X[med], J, None)
        f = np.random.RandomState(s).choice(len(X), 5, replace=False)
        lab, C, J, buang = trimmed_kmeans(X, X[f], 0.05)
        if "trimmed, Forgy" not in hasil or J < hasil["trimmed, Forgy"][2]:
            hasil["trimmed, Forgy"] = (lab, C, J, buang)
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.95), sharey=True)
    dalam = (np.abs(X[:, 0] - 4.5) < 9) & (np.abs(X[:, 1] - 3) < 8)
    for s, (nama, (lab, C, J, buang)) in zip(ax, hasil.items()):
        tampil = dalam if buang is None else dalam & ~buang
        s.scatter(X[tampil, 0], X[tampil, 1], s=1.2,
                  c=_warna(lab[tampil]), lw=0)
        if buang is not None:
            b = dalam & buang
            s.scatter(X[b, 0], X[b, 1], s=5, marker="x", lw=0.5,
                      color=ABU)
        ok = (np.abs(C[:, 0] - 4.5) < 9) & (np.abs(C[:, 1] - 3) < 8)
        s.scatter(*C[ok].T, marker="x", s=16, lw=1.0, color="black")
        s.set_title(f"{nama}\n{int((~ok).sum())} pusat di luar gambar",
                    fontsize=6.3)
        s.set_xlim(-4.5, 13.5)
        s.set_ylim(-5, 11)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab13-pencilan")


def bab13_bola():
    from sklearn.cluster import KMeans
    from bab13_bola import arah, spherical_kmeans
    X, y = arah(2)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2), sharey=True)
    lab1 = KMeans(3, n_init=10, random_state=BENIH).fit(X).labels_
    lab2 = max((spherical_kmeans(X, X[np.random.RandomState(s)
                .choice(len(X), 3, replace=False)]) for s in range(10)),
               key=lambda h: h[2])[0]
    for s, lab, judul in ((ax[0], lab1, "K-means"),
                          (ax[1], lab2, "spherical K-means")):
        s.scatter(X[:, 0], X[:, 1], s=1.5, c=_warna(lab), lw=0)
        s.plot(0, 0, "+", color="black", ms=6)
        s.set_title(judul, fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.5)
    simpan(fig, "bab13-bola")


# =====================================================================
#  Bab 14 -- Kernel K-means dan spectral clustering
# =====================================================================
def bab14_angkat():
    from bab11_data import enam_data
    _, X, y = enam_data()[5]
    r2 = (X ** 2).sum(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.0))
    ax[0].scatter(X[:, 0], X[:, 1], s=1.5, c=_warna(y), lw=0)
    ax[0].set_aspect("equal")
    ax[0].set_title("bidang asal", fontsize=7)
    ax[1].scatter(X[:, 0], r2, s=1.5, c=_warna(y), lw=0)
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_1^2 + x_2^2$")
    ax[1].set_title("sesudah diangkat", fontsize=7)
    for s in ax:
        _rapikan(s)
    fig.tight_layout(w_pad=0.8)
    simpan(fig, "bab14-angkat")


def bab14_kernel():
    from sklearn.metrics import adjusted_rand_score as ari
    from bab11_data import enam_data
    from bab14_kernel import rbf, terbaik
    data = {n: (X, y) for n, X, y in enam_data()}
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.0))
    for s, (nama, g) in zip(ax, (("bulan", 5), ("cincin", 20))):
        X, y = data[nama]
        (lab, _), _ = terbaik(rbf(X, g), 2)
        s.scatter(X[:, 0], X[:, 1], s=1.5, c=_warna(lab), lw=0)
        s.set_title(rf"{nama}, $\gamma = {g}$, ARI "
                    f"{angka(round(ari(y, lab), 2) + 0.0, 2)}",
                    fontsize=6.5)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.6)
    simpan(fig, "bab14-kernel")


def bab14_spektral():
    from sklearn.cluster import KMeans
    from bab11_data import enam_data
    from bab14_kernel import rbf
    _, X, y = enam_data()[5]
    W = rbf(X, 20)
    np.fill_diagonal(W, 0)
    d = W.sum(axis=1)
    L = np.eye(len(X)) - W / np.sqrt(np.outer(d, d))
    nilai, vektor = np.linalg.eigh(L)
    U = vektor[:, :2]
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    lab = KMeans(2, n_init=10, random_state=BENIH).fit(U).labels_
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.75))
    ax[0].plot(np.arange(1, 9), nilai[:8], "o", ms=3, color=BIRU)
    ax[0].set_xlabel("urutan")
    ax[0].set_title("nilai eigen terkecil", fontsize=6.5)
    ax[1].scatter(U[:, 0], U[:, 1], s=3, c=_warna(y), lw=0)
    ax[1].set_title("baris $\\mathbf{U}$ (label benar)", fontsize=6.5)
    ax[1].set_aspect("equal", adjustable="datalim")
    ax[2].scatter(X[:, 0], X[:, 1], s=1.5, c=_warna(lab), lw=0)
    ax[2].set_title("hasil spectral", fontsize=6.5)
    ax[2].set_aspect("equal")
    for s in ax:
        _rapikan(s)
    fig.tight_layout(w_pad=0.4)
    simpan(fig, "bab14-spektral")


def bab14_gamma():
    from sklearn.metrics import adjusted_rand_score as ari
    from bab11_data import enam_data
    from bab14_spektral import spektral
    data = {n: (X, y) for n, X, y in enam_data()}
    gs = np.logspace(-1, 2, 13)
    fig, ax = plt.subplots(figsize=(4.7, 1.8))
    for nama, w in (("gumpalan", BIRU), ("bulan", JINGGA),
                    ("cincin", HIJAU)):
        X, y = data[nama]
        v = [ari(y, spektral(X, y.max() + 1, g)[0]) for g in gs]
        ax.semilogx(gs, v, "-o", ms=2.5, lw=0.9, color=w, label=nama)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("ARI")
    ax.legend(fontsize=5.5)
    kunci_label(ax, "x")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab14-gamma")


# =====================================================================
#  Bab 15 -- Mempercepat Lloyd: Elkan dan Hamerly
# =====================================================================
def bab15_segitiga():
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    x, c, c2 = np.array([0.6, 0.5]), np.array([0.0, 0.0]), \
        np.array([3.2, 0.9])
    u = np.hypot(*(x - c))
    ax.add_patch(Circle(c, 2 * u, fill=False, ls="--", color=ABU,
                        lw=0.7))
    ax.add_patch(Circle(x, u, fill=False, color=JINGGA, lw=0.8))
    for p, nama, w in ((x, r"$\mathbf{x}$", "black"),
                       (c, r"$\boldsymbol{\mu}_a$", BIRU),
                       (c2, r"$\boldsymbol{\mu}_b$", MERAH)):
        ax.plot(*p, "o", ms=4, color=w)
        ax.annotate(nama, p, xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=w)
    ax.annotate("", xy=x, xytext=c,
                arrowprops=dict(arrowstyle="-", color=JINGGA, lw=0.8))
    ax.text(0.12, 0.38, "$u$", fontsize=7, color=JINGGA)
    ax.text(1.4, -1.35, r"lingkaran berjari-jari $2u$" + "\n" +
            r"di sekitar $\boldsymbol{\mu}_a$", fontsize=6, color=ABU)
    ax.set_xlim(-1.8, 3.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    simpan(fig, "bab15-segitiga")


def bab15_periterasi():
    from sklearn.cluster import kmeans_plusplus
    from bab15_cepat import elkan, hamerly, lloyd_hitung
    from bab15_cocok import data_uji
    data = {n: (X, K) for n, X, K in data_uji()}
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9), sharey=False)
    for s, nama in zip(ax, ("blobs 20k", "acak d=50")):
        X, K = data[nama]
        C0, _ = kmeans_plusplus(X, K, random_state=0)
        for f, w, lab in ((lloyd_hitung, ABU, "Lloyd"),
                          (elkan, BIRU, "Elkan"),
                          (hamerly, JINGGA, "Hamerly")):
            c = np.array(f(X, C0)[3])
            s.semilogy(np.arange(1, len(c) + 1), c, "-", lw=0.9,
                       color=w, label=lab)
        s.set_title(nama, fontsize=7)
        s.set_xlabel("iterasi")
        _rapikan(s)
    ax[0].set_ylabel("jarak dihitung")
    ax[0].legend(fontsize=5.5)
    fig.tight_layout(w_pad=0.8)
    simpan(fig, "bab15-periterasi")


def bab15_waktu():
    import time
    from sklearn.cluster import KMeans, kmeans_plusplus
    from bab15_cepat import elkan, hamerly, lloyd_hitung
    from bab15_cocok import data_uji

    def ukur(f):
        terbaik = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            f()
            terbaik = min(terbaik, time.perf_counter() - t0)
        return terbaik

    nama_data, hasil = [], []
    for nama, X, K in data_uji()[3:]:
        C0, _ = kmeans_plusplus(X, K, random_state=0)
        hasil.append([ukur(lambda: g(X, C0)) for g in
                      (lloyd_hitung, elkan, hamerly)] +
                     [ukur(lambda: KMeans(K, init=C0, n_init=1, tol=0,
                                          algorithm=a).fit(X))
                      for a in ("lloyd", "elkan")])
        nama_data.append(nama)
    hasil = np.array(hasil)
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    lebar = 0.16
    for j, (lab, w) in enumerate((("Lloyd (NumPy)", ABU),
                                  ("Elkan (NumPy)", BIRU),
                                  ("Hamerly (NumPy)", JINGGA),
                                  ("sklearn lloyd", HIJAU),
                                  ("sklearn elkan", MERAH))):
        ax.bar(np.arange(len(nama_data)) + (j - 2) * lebar, hasil[:, j],
               lebar, color=w, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(range(len(nama_data)))
    ax.set_xticklabels(nama_data)
    ax.set_ylabel("waktu (detik)")
    ax.set_ylim(0.01, 40)
    ax.legend(fontsize=5, ncol=3, loc="upper center")
    kunci_label(ax, "x", "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab15-waktu")
    for n, h in zip(nama_data, hasil):
        print("  ", n, " ".join(f"{v:.3f}" for v in h))


# =====================================================================
#  Bab 16 -- Mini-batch dan online K-means
# =====================================================================
def bab16_jalur():
    from sklearn.cluster import kmeans_plusplus
    from bab04_data import gumpalan
    X, _ = gumpalan(20000)
    C0, _ = kmeans_plusplus(X, 5, random_state=0)
    urutan = np.random.default_rng(BENIH).permutation(len(X))[:3000]
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.1), sharey=True)
    for s, eta, judul in ((ax[0], None, r"langkah $1/n_k$"),
                          (ax[1], 0.1, r"langkah tetap $0{,}1$")):
        C = C0.astype(float).copy()
        n_k = np.zeros(5)
        jalur = [C.copy()]
        for i in urutan:
            k = ((C - X[i]) ** 2).sum(axis=1).argmin()
            n_k[k] += 1
            C[k] += (X[i] - C[k]) * (1 / n_k[k] if eta is None else eta)
            jalur.append(C.copy())
        jalur = np.array(jalur)
        s.scatter(X[:4000, 0], X[:4000, 1], s=0.5, color=ABU_GARIS, lw=0)
        for k in range(5):
            s.plot(jalur[:, k, 0], jalur[:, k, 1], "-", lw=0.5,
                   color=WARNA_K[k])
            s.plot(*jalur[-1, k], "x", ms=4, color="black", mew=1.0)
        s.set_title(judul, fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.5)
    simpan(fig, "bab16-jalur")


def bab16_langkah():
    from sklearn.cluster import kmeans_plusplus
    from bab04_data import gumpalan
    from bab05_lloyd import lloyd
    from bab16_cocok import inersia
    X, _ = gumpalan(20000)
    C0, _ = kmeans_plusplus(X, 5, random_state=0)
    C_l, _, _ = lloyd(X, C0)
    J_l = inersia(X, C_l)
    urutan = np.concatenate([np.random.default_rng(BENIH + p)
                             .permutation(len(X)) for p in range(3)])
    fig, ax = plt.subplots(figsize=(4.7, 1.9))
    for eta, w, lab in ((None, BIRU, r"$1/n_k$"), (0.001, HIJAU, "0,001"),
                        (0.01, JINGGA, "0,01"), (0.1, MERAH, "0,1")):
        C = C0.astype(float).copy()
        n_k = np.zeros(5)
        t_, v = [], []
        for j, i in enumerate(urutan, 1):
            k = ((C - X[i]) ** 2).sum(axis=1).argmin()
            n_k[k] += 1
            C[k] += (X[i] - C[k]) * (1 / n_k[k] if eta is None else eta)
            if j % 500 == 0:
                t_.append(j)
                v.append(inersia(X, C) / J_l - 1)
        ax.loglog(t_, np.maximum(v, 1e-8), color=w, lw=0.9, label=lab)
    ax.set_xlabel("titik yang sudah dilihat")
    ax.set_ylabel(r"$J/J_{\mathrm{Lloyd}} - 1$")
    ax.legend(fontsize=5.5, title="ukuran langkah", title_fontsize=5.5)
    kunci_label(ax, "x", "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab16-langkah")


def bab16_waktu():
    import time
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from bab16_besar import data_besar
    X = data_besar()
    hasil = []
    for s in range(5):
        t0 = time.perf_counter()
        J = KMeans(50, n_init=1, random_state=s).fit(X).inertia_
        hasil.append(("KMeans", time.perf_counter() - t0, J))
        for b in (256, 1024, 4096):
            t0 = time.perf_counter()
            mb = MiniBatchKMeans(50, batch_size=b, n_init=1,
                                 random_state=s).fit(X)
            hasil.append((f"mini-batch {b}", time.perf_counter() - t0,
                          -mb.score(X)))
    J_min = min(h[2] for h in hasil)
    fig, ax = plt.subplots(figsize=(4.7, 2.0))
    for nama, w, m in (("KMeans", ABU, "s"), ("mini-batch 256", BIRU, "o"),
                       ("mini-batch 1024", JINGGA, "^"),
                       ("mini-batch 4096", HIJAU, "D")):
        tt = [h[1] for h in hasil if h[0] == nama]
        jj = [h[2] / J_min - 1 for h in hasil if h[0] == nama]
        ax.scatter(tt, jj, s=14, marker=m, color=w, label=nama, lw=0)
        print("  ", nama, "median waktu", f"{np.median(tt):.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("waktu (detik)")
    ax.set_ylabel(r"$J/J^* - 1$")
    ax.legend(fontsize=5.5)
    kunci_label(ax, "x")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab16-waktu")


# =====================================================================
#  Bab 17 -- Fuzzy c-means, bisecting K-means, dan klaster seimbang
# =====================================================================
def bab17_fuzzy():
    from sklearn.cluster import KMeans
    from bab04_data import gumpalan
    from bab17_fuzzy import fcm
    X, _ = gumpalan()
    C0 = KMeans(5, n_init=10, random_state=BENIH).fit(X).cluster_centers_
    gx, gy = np.meshgrid(np.linspace(-3, 12, 300), np.linspace(-3, 9, 240))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.7), sharey=True)
    for s, m in zip(ax, (1.5, 2, 5)):
        _, C, _ = fcm(X, C0, m)
        d2 = ((G[:, None] - C[None]) ** 2).sum(axis=2)
        w = d2 ** (-1 / (m - 1))
        U = (w / w.sum(axis=1, keepdims=True)).max(axis=1)
        im = s.imshow(U.reshape(gx.shape), origin="lower", cmap="Blues",
                      vmin=0.2, vmax=1, extent=(-3, 12, -3, 9),
                      aspect="auto")
        s.scatter(X[:, 0], X[:, 1], s=0.4, color=ABU, lw=0)
        s.scatter(*C.T, marker="x", s=12, lw=0.9, color=MERAH)
        s.set_title(rf"$m = {angka_mat(m, 1) if m != int(m) else int(m)}$",
                    fontsize=7)
        s.set_aspect("equal")
        _rapikan(s)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("keanggotaan terbesar", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    simpan(fig, "bab17-fuzzy")


def bab17_bisecting():
    from bab04_data import gumpalan
    from bab17_bisecting import bisecting
    X, _ = gumpalan()
    fig, ax = plt.subplots(1, 4, figsize=(4.7, 1.45), sharey=True)
    for s, K in zip(ax, (2, 3, 4, 5)):
        lab, _ = bisecting(X, K, 0)
        s.scatter(X[:, 0], X[:, 1], s=0.8, c=_warna(lab), lw=0)
        s.set_title(f"$K = {K}$", fontsize=7)
        s.set_aspect("equal")
        s.set_xticks([])
        s.set_yticks([])
        for sisi in s.spines.values():
            sisi.set_color(ABU_GARIS)
    fig.tight_layout(w_pad=0.3)
    simpan(fig, "bab17-bisecting")


def bab17_seimbang():
    from sklearn.cluster import KMeans
    from bab17_seimbang import seimbang
    rng = np.random.default_rng(BENIH)
    X = rng.uniform(0, 10, (600, 2))
    km = KMeans(6, n_init=10, random_state=BENIH).fit(X)
    lab, C, _ = seimbang(X, km.cluster_centers_)
    warna6 = WARNA_K + ["#8A6D3B"]
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2), sharey=True)
    for s, l, CC, judul in ((ax[0], km.labels_, km.cluster_centers_,
                             "K-means"), (ax[1], lab, C, "seimbang")):
        s.scatter(X[:, 0], X[:, 1], s=2, c=[warna6[k] for k in l], lw=0)
        s.scatter(*CC.T, marker="x", s=16, lw=1.0, color="black")
        uk = sorted(np.bincount(l).tolist())
        teks = f"semua {uk[0]} titik" if min(uk) == max(uk) \
            else f"{min(uk)} sampai {max(uk)} titik"
        s.set_title(f"{judul}: {teks}", fontsize=6.5)
        s.set_aspect("equal")
        _rapikan(s)
    fig.tight_layout(w_pad=0.5)
    simpan(fig, "bab17-seimbang")


# =====================================================================
#  Bab 18 -- Kuantisasi vektor dan kompresi gambar
# =====================================================================
def bab18_lloydmax():
    from scipy.stats import norm
    from bab18_lloydmax import lloyd_max
    c8, t8, _, _ = lloyd_max(8)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 1.9))
    x = np.linspace(-3.5, 3.5, 400)
    ax[0].plot(x, norm.pdf(x), color=BIRU, lw=1)
    for a in t8[1:-1]:
        ax[0].axvline(a, color=ABU_GARIS, lw=0.6, ls="--")
    ax[0].scatter(c8, np.zeros(8), marker="x", s=18, lw=1.0,
                  color=JINGGA, zorder=3, clip_on=False)
    ax[0].set_title("K = 8: ambang (garis) dan level (silang)",
                    fontsize=6.5)
    ax[0].set_xlabel("x")
    ax[0].set_ylim(0, 0.43)
    c32 = lloyd_max(32)[0]
    j = np.arange(32)
    u = (j + 0.5) / 32
    ax[1].plot(j, np.sqrt(3) * norm.ppf(u), color=HIJAU, lw=0.9,
               label=r"kerapatan $\propto \varphi^{1/3}$")
    ax[1].plot(j, norm.ppf(u), color=ABU, lw=0.9, ls="--",
               label=r"kerapatan $\propto \varphi$")
    ax[1].scatter(j, c32, s=5, color=JINGGA, zorder=3,
                  label="Lloyd-Max")
    ax[1].set_title("K = 32: letak level", fontsize=6.5)
    ax[1].set_xlabel("urutan level")
    ax[1].legend(fontsize=5.5, loc="upper left")
    for s in ax:
        _rapikan(s)
    fig.tight_layout(w_pad=1.0)
    simpan(fig, "bab18-lloydmax")


def bab18_laju():
    from bab18_laju import DAFTAR_K, kurva
    rng = np.random.default_rng(BENIH)
    fig, ax = plt.subplots(figsize=(4.7, 2.1))
    K = np.array(DAFTAR_K)
    warna = [BIRU, JINGGA, HIJAU, MERAH, ABU, "#8A6D3B"]
    for d, w in zip((1, 2, 3, 5, 10, 20), warna):
        _, uji = kurva(d, rng)
        ax.plot(K, uji / uji[0], "o-", ms=2.5, lw=0.9, color=w,
                label=f"d = {d}")
        ax.plot(K, (K / 4.0) ** (-2 / d), ls=":", lw=0.7, color=w)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(K)
    ax.set_xticklabels([str(k) for k in K])
    ax.set_xlabel("K")
    ax.set_ylabel("D(K) / D(4), data uji")
    ax.legend(fontsize=5.5, ncol=2, loc="lower left")
    kunci_label(ax, "x", "y")
    _rapikan(ax)
    fig.tight_layout()
    simpan(fig, "bab18-laju")


def bab18_sarang():
    from matplotlib.patches import Polygon
    from scipy.spatial import Voronoi
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(BENIH)
    X = rng.uniform(size=(400_000, 2))
    C = KMeans(256, n_init=1, random_state=BENIH).fit(X).cluster_centers_
    v = Voronoi(C)
    warna = {5: BIRU_MUDA, 6: "white", 7: JINGGA_MUDA}
    fig, ax = plt.subplots(figsize=(2.9, 2.9))
    for r in v.point_region:
        reg = v.regions[r]
        if -1 in reg:
            continue
        P = v.vertices[reg]
        if (P < 0).any() or (P > 1).any():
            continue
        ax.add_patch(Polygon(P, closed=True, lw=0.4, ec=ABU,
                             fc=warna.get(len(reg), MERAH_MUDA)))
    ax.scatter(*C.T, s=1.5, color=BIRU, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sisi in ax.spines.values():
        sisi.set_color(ABU_GARIS)
    fig.tight_layout()
    simpan(fig, "bab18-sarang")


def bab18_warna():
    from sklearn.datasets import load_sample_image
    from bab18_warna import kuantisasi_warna
    gambar = load_sample_image("china.jpg")
    X = gambar.reshape(-1, 3).astype(float)
    fig, ax = plt.subplots(2, 3, figsize=(4.7, 2.45))
    panel = [(gambar, "asli, 24 bit per piksel")]
    for K in (4, 16, 64, 256):
        palet, label = kuantisasi_warna(gambar, K)
        hasil = palet[label].reshape(gambar.shape)
        panel.append((hasil.round().astype(np.uint8),
                      f"K = {K}, {int(np.log2(K))} bit per piksel"))
        if K == 16:
            galat = np.sqrt(((X - palet[label]) ** 2).sum(axis=1))
    for s, (g, judul) in zip(ax.flat, panel):
        s.imshow(g)
        s.set_title(judul, fontsize=5.5)
    ax.flat[5].imshow(galat.reshape(gambar.shape[:2]), cmap="Greys",
                      vmin=0, vmax=80)
    ax.flat[5].set_title("K = 16: jarak ke warna palet", fontsize=5.5)
    for s in ax.flat:
        s.set_xticks([])
        s.set_yticks([])
        for sisi in s.spines.values():
            sisi.set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=1.2)
    simpan(fig, "bab18-warna")


# =====================================================================
#  Bab 19 -- Studi kasus: segmentasi pelanggan Online Retail
# =====================================================================
def _rfm19():
    from bab19_data import baca, bersihkan, rfm
    return rfm(*bersihkan(baca()))


def bab19_sebaran():
    tabel = _rfm19()
    fig, ax = plt.subplots(2, 3, figsize=(4.7, 2.7))
    judul = ("R (hari sejak beli terakhir)", "F (banyaknya faktur)",
             "M (belanja bersih, pound)")
    for j, v in enumerate("RFM"):
        x = tabel[v].to_numpy(float)
        ax[0, j].hist(x, bins=40, color=BIRU, lw=0)
        ax[0, j].set_title(judul[j], fontsize=6)
        lx = np.log1p(x) if v == "R" else np.log(x)
        ax[1, j].hist(lx, bins=40, color=HIJAU, lw=0)
        ax[1, j].set_title(("log(1 + R)", "log F", "log M")[j],
                           fontsize=6)
        for s in ax[:, j]:
            s.tick_params(labelsize=5.5)
            s.set_yticks([])
            _rapikan(s)
            s.spines["left"].set_visible(False)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    simpan(fig, "bab19-sebaran")


def bab19_pilihk():
    from sklearn.cluster import KMeans
    from sklearn.metrics import (calinski_harabasz_score,
                                 davies_bouldin_score, silhouette_score)
    from sklearn.mixture import GaussianMixture
    from bab09_pilih import kestabilan
    from bab19_data import fitur_log
    Z = fitur_log(_rfm19())
    Ks = np.arange(2, 11)
    hasil = {k: [] for k in ("inersia", "silhouette", "Calinski-Harabasz",
                             "Davies-Bouldin", "kestabilan",
                             "BIC campuran Gaussian")}
    for K in Ks:
        km = KMeans(K, n_init=10, random_state=BENIH).fit(Z)
        gm = GaussianMixture(K, covariance_type="full", n_init=3,
                             random_state=BENIH).fit(Z)
        for nama, v in zip(hasil, (km.inertia_,
                                   silhouette_score(Z, km.labels_),
                                   calinski_harabasz_score(Z, km.labels_),
                                   davies_bouldin_score(Z, km.labels_),
                                   kestabilan(Z, K), gm.bic(Z))):
            hasil[nama].append(v)
    terbaik = {"silhouette": np.argmax, "Calinski-Harabasz": np.argmax,
               "Davies-Bouldin": np.argmin, "kestabilan": np.argmax,
               "BIC campuran Gaussian": np.argmin}
    fig, ax = plt.subplots(2, 3, figsize=(4.7, 2.6), sharex=True)
    for s, (nama, v) in zip(ax.flat, hasil.items()):
        s.plot(Ks, v, "o-", ms=2.5, lw=0.9, color=BIRU)
        if nama in terbaik:
            i = terbaik[nama](v)
            s.plot(Ks[i], v[i], "o", ms=5, mfc="none", color=JINGGA)
        s.set_title(nama, fontsize=6.5)
        s.tick_params(labelsize=5.5)
        s.set_xticks(Ks)
        _rapikan(s)
    for s in ax[1]:
        s.set_xlabel("K", fontsize=6.5)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    simpan(fig, "bab19-pilihk")


def _sumbu_log(s, x, y):
    s.set_xscale("log")
    s.set_yscale("log")
    s.set_xlabel(x, fontsize=6.5)
    s.set_ylabel(y, fontsize=6.5)
    s.tick_params(labelsize=5.5)
    _rapikan(s)


def bab19_segmen():
    from bab19_segmen import segmen
    tabel = _rfm19()
    label, _ = segmen(tabel)
    rng = np.random.default_rng(BENIH)
    goyang = np.exp(rng.uniform(-0.15, 0.15, len(tabel)))
    warna = [ABU, HIJAU, JINGGA, BIRU]
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2))
    for k in range(4):
        m = label == k
        ax[0].scatter(tabel.F[m] * goyang[m], tabel.M[m], s=1.2,
                      color=warna[k], lw=0, label=f"segmen {k + 1}")
        ax[1].scatter(tabel.R[m] + 1, tabel.M[m], s=1.2,
                      color=warna[k], lw=0)
    _sumbu_log(ax[0], "F (faktur, digoyang sedikit)", "M (pound)")
    _sumbu_log(ax[1], "1 + R (hari)", "M (pound)")
    ax[0].legend(fontsize=5.5, markerscale=4, loc="lower right")
    kunci_label(ax[0], "x", "y")
    kunci_label(ax[1], "x", "y")
    fig.tight_layout(w_pad=1.0)
    simpan(fig, "bab19-segmen")


def bab19_gmm():
    from sklearn.mixture import GaussianMixture
    from bab19_data import fitur_log
    tabel = _rfm19()
    gm = GaussianMixture(6, covariance_type="full", n_init=3,
                         random_state=BENIH).fit(fitur_log(tabel))
    lab = gm.predict(fitur_log(tabel))
    rng = np.random.default_rng(BENIH)
    goyang = np.exp(rng.uniform(-0.15, 0.15, len(tabel)))
    warna = [BIRU, JINGGA, HIJAU, MERAH, ABU, "#8A6D3B"]
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for k in range(6):
        m = lab == k
        ax.scatter(tabel.F[m] * goyang[m], tabel.M[m], s=1.2,
                   color=warna[k], lw=0)
    _sumbu_log(ax, "F (faktur, digoyang sedikit)", "M (pound)")
    kunci_label(ax, "x", "y")
    fig.tight_layout()
    simpan(fig, "bab19-gmm")


# =====================================================================
#  Bab 20 -- Ke mana setelah ini
# =====================================================================
def bab20_hdbscan():
    from sklearn.cluster import HDBSCAN
    from bab19_data import fitur_log
    tabel = _rfm19()
    rng = np.random.default_rng(BENIH)
    goyang = tabel.assign(F=tabel.F + rng.uniform(-0.5, 0.5, len(tabel)))
    lab_a = HDBSCAN(min_cluster_size=50).fit(fitur_log(tabel)).labels_
    lab_b = HDBSCAN(min_cluster_size=50).fit(fitur_log(goyang)).labels_
    warna = [BIRU, JINGGA, HIJAU, MERAH, "#8A6D3B"]
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.2), sharey=True)
    for s, T, lab, judul in ((ax[0], tabel, lab_a, "F bilangan bulat"),
                             (ax[1], goyang, lab_b, "F digoyang")):
        g = np.exp(rng.uniform(-0.12, 0.12, len(T))) if s is ax[0] \
            else np.ones(len(T))
        m = lab < 0
        s.scatter(T.F[m] * g[m], T.M[m], s=1.0, color=ABU_GARIS, lw=0)
        for k in range(lab.max() + 1):
            m = lab == k
            s.scatter(T.F[m] * g[m], T.M[m], s=1.2, color=warna[k % 5],
                      lw=0)
        s.set_title(f"{judul}: {lab.max() + 1} klaster, "
                    f"noise {angka((lab < 0).mean() * 100, 0)}%",
                    fontsize=6.5)
        _sumbu_log(s, "F (faktur)", "M (pound)" if s is ax[0] else "")
        kunci_label(s, "x", "y")
    fig.tight_layout(w_pad=0.8)
    simpan(fig, "bab20-hdbscan")


def bab20_coreset():
    from bab20_coreset import coreset, data_coreset
    X = data_coreset()
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3), sharex=True,
                           sharey=True)
    S = X[rng.choice(len(X), 300, replace=False)]
    jauh = (S[:, 0] > 30).sum()
    ax[0].scatter(*S.T, s=2, color=BIRU, lw=0)
    ax[0].set_title(f"sampel seragam: {jauh} dari 300 titik\n"
                    "di gumpalan kecil", fontsize=6.5)
    S, w = coreset(X, 300, rng)
    jauh = (S[:, 0] > 30).sum()
    ax[1].scatter(*S.T, s=2, color=JINGGA, lw=0)
    ax[1].set_title(f"coreset: {jauh} dari 300 titik\n"
                    "di gumpalan kecil", fontsize=6.5)
    for s in ax:
        s.annotate("gumpalan kecil\n(500 titik)", (40, 40),
                   xytext=(22, 32), fontsize=5.5, color=ABU,
                   arrowprops=dict(arrowstyle="->", color=ABU, lw=0.5))
        s.set_aspect("equal")
        s.tick_params(labelsize=5.5)
        _rapikan(s)
    fig.tight_layout(w_pad=0.8)
    simpan(fig, "bab20-coreset")


def bab20_tsne():
    import warnings
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE
    warnings.filterwarnings("ignore", category=FutureWarning)
    X, y = load_digits(return_X_y=True)
    T = TSNE(2, init="pca", random_state=0).fit_transform(X)
    U = np.random.default_rng(BENIH).uniform(size=(1500, 10))
    TU = TSNE(2, init="pca", random_state=0).fit_transform(U)
    lab = KMeans(10, n_init=10, random_state=BENIH).fit(TU).labels_
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(4.7, 2.3))
    ax[0].scatter(*T.T, s=1.2, c=[cmap(v) for v in y], lw=0)
    ax[0].set_title("digits, warna = angka sebenarnya", fontsize=6.5)
    ax[1].scatter(*TU.T, s=1.2, c=[cmap(v) for v in lab], lw=0)
    ax[1].set_title("seragam 10 dimensi, warna = K-means", fontsize=6.5)
    for s in ax:
        s.set_xticks([])
        s.set_yticks([])
        s.set_aspect("equal")
        for sisi in s.spines.values():
            sisi.set_color(ABU_GARIS)
    fig.tight_layout(w_pad=0.8)
    simpan(fig, "bab20-tsne")


if __name__ == "__main__":
    pola = re.compile(r"^bab\d\d_")
    pilihan = sys.argv[1:]
    fungsi = [(k, v) for k, v in sorted(globals().items())
              if pola.match(k) and callable(v)]
    if pilihan:
        fungsi = [(k, v) for k, v in fungsi
                  if any(p in k for p in pilihan)]
    if not fungsi:
        print("tidak ada gambar yang cocok")
    for _, f in fungsi:
        f()
