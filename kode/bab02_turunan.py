"""Bab 2: memeriksa perkakas matematika dengan komputer.

(1) jabaran kuadrat jarak, (2) gradien f(a) = sum ||x_i - a||^2 lawan
beda hingga, (3) gradient descent dengan beberapa ukuran langkah lawan
satu langkah Newton, (4) median meminimumkan jumlah jarak mutlak,
(5) kuadrat jarak tidak memenuhi ketaksamaan segitiga.
Data: 500 titik gumpalan dari bab04_data.
"""
import numpy as np

from bab04_data import BENIH, gumpalan


def f(X, a):
    return ((X - a) ** 2).sum()


def gradien(X, a):
    return 2 * len(X) * (a - X.mean(axis=0))


def gradient_descent(X, a, eta, tol=1e-10, maks_iter=10_000):
    rata = X.mean(axis=0)
    for t in range(1, maks_iter + 1):
        a = a - eta * gradien(X, a)
        jarak = np.linalg.norm(a - rata)
        if jarak < tol or jarak > 1e6:
            return a, t
    return a, maks_iter


if __name__ == "__main__":
    rng = np.random.default_rng(BENIH)
    x, m = rng.normal(size=5), rng.normal(size=5)
    kiri = ((x - m) ** 2).sum()
    kanan = x @ x - 2 * x @ m + m @ m
    print(f"(1) ||x - m||^2 = {kiri:.12f}")
    print(f"    jabaran     = {kanan:.12f}")

    X, _ = gumpalan()
    n = len(X)
    a = np.array([10.0, -3.0])
    h = 1e-6
    beda = np.array([(f(X, a + h * e) - f(X, a - h * e)) / (2 * h)
                     for e in np.eye(2)])
    print(f"(2) gradien rumus      = {np.round(gradien(X, a), 4)}")
    print(f"    gradien beda hingga = {np.round(beda, 4)}")

    print("(3) gradient descent dari a = (10, -3), n = 500")
    print("    2*eta*n   iterasi   jarak akhir ke rata-rata")
    for q in (0.1, 0.5, 1.0, 1.5, 1.9, 2.1):
        b, t = gradient_descent(X, a, q / (2 * n))
        print(f"    {q:7.1f} {t:9d}   {np.linalg.norm(b - X.mean(0)):.2e}")
    newton = a - gradien(X, a) / (2 * n)        # Hessian = 2n I
    print(f"    Newton, satu langkah: jarak "
          f"{np.linalg.norm(newton - X.mean(0)):.2e}")

    y = X[:, 0]
    kisi = np.linspace(y.min(), y.max(), 20001)
    mutlak = np.abs(y[:, None] - kisi[None, :]).sum(axis=0)
    kuadrat = ((y[:, None] - kisi[None, :]) ** 2).sum(axis=0)
    ys = np.sort(y)
    print(f"(4) jumlah |y - a| minimum pada kisi: {mutlak.min():.4f}")
    print(f"    jumlah |y - a| di median:         "
          f"{np.abs(y - np.median(y)).sum():.4f}")
    print(f"    datum ke-250 dan ke-251: {ys[249]:.4f}, {ys[250]:.4f}")
    print(f"    jumlah (y - a)^2 minimum di a = "
          f"{kisi[kuadrat.argmin()]:.3f}; rata-rata = {y.mean():.3f}")

    p, q, r = np.array([0.0]), np.array([1.0]), np.array([2.0])
    d = lambda u, v: np.abs(u - v).item()
    print(f"(5) d(p,r) = {d(p, r):.0f} <= d(p,q) + d(q,r) = "
          f"{d(p, q) + d(q, r):.0f}")
    print(f"    d(p,r)^2 = {d(p, r) ** 2:.0f} > d(p,q)^2 + d(q,r)^2 = "
          f"{d(p, q) ** 2 + d(q, r) ** 2:.0f}")
