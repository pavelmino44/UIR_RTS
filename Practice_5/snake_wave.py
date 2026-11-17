import numpy as np
import matplotlib.pyplot as plt

# ---------- Параметры модели ----------
N      = 12          # число звеньев
L      = 1.0         # длина звена
A      = 0.6         # амплитуда волны, рад
omega  = 2.0         # круговая частота, рад/с
delta  = np.pi / 3   # фазовый сдвиг между звеньями, рад
phi0   = np.pi / 2   # начальная ориентация, рад

# ---------- Волна ----------
def angles(t, N=N, A=A, omega=omega, delta=delta):
    """Возвращает массив углов в сочленениях для времени t."""
    i = np.arange(N)
    return A * np.sin(omega * t - i * delta)

# ---------- Прямая кинематика ----------
def forward_kinematics(thetas, L=L, phi0=phi0, start=(0.0, 0.0)):
    """По массиву углов в сочленениях вернуть координаты узлов цепочки."""
    pts = [np.array(start, dtype=float)]
    phi = phi0
    for th in thetas:
        phi += th
        pts.append(pts[-1] + L * np.array([np.cos(phi), np.sin(phi)]))
    return np.array(pts)

# ---------- Визуализация ----------
def plot_snapshots(times, ax):
    for t in times:
        pts = forward_kinematics(angles(t))
        # приподнимаем каждую змею по оси Y, чтобы формы не накладывались
        ax.plot(pts[:, 0], pts[:, 1] + 2.0 * times.index(t),
                '-o', ms=3, lw=1.5, label=f't = {t:.2f} c')
    # --- окончание листинга (в методичке обрезано) ---
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Форма змеи при бегущей волне (N = {N})')
    ax.legend(loc='upper left', fontsize=8)


# Запуск только при прямом вызове файла, чтобы при импорте
# из других скриптов (задание 2) окно не открывалось.
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(8, 8))
    times = [0.0, 0.4, 0.8, 1.2, 1.6]     # список: в plot_snapshots используется times.index(t)
    plot_snapshots(times, ax)
    plt.tight_layout()
    plt.show()
