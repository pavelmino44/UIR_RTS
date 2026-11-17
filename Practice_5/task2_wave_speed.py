"""
Задание 2 (пункты 2–3).
  2.2 — сохранение анимации змеи в GIF (snake_wave.gif);
  2.3 — измерение скорости бегущей волны вдоль тела.
Использует функции angles() и forward_kinematics() из snake_wave.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from snake_wave import angles, forward_kinematics, N, L, omega, delta

os.makedirs('figures', exist_ok=True)
DT = 0.05                                    # шаг времени кадра, как в исходной анимации


# ---------- П.2. GIF-файл ----------
def crest_joints(t, N_=N, omega_=omega, delta_=delta):
    """Номера сочленений, ближайших к гребням волны (θ = +A).
    Гребень: ωt − iΔφ = π/2 + 2πk  →  i = (ωt − π/2 − 2πk)/Δφ.
    На теле может быть несколько гребней — по одному на каждую длину волны."""
    lam = 2 * np.pi / delta_                       # длина волны в звеньях
    i0 = ((omega_ * t - np.pi / 2) / delta_) % lam
    ks = [int(round(i)) for i in np.arange(i0 - lam, N_, lam)]
    return [k for k in ks if 0 <= k < N_]


def save_gif(filename='snake_wave.gif'):
    """Анимация как в snake_animation.py + красные точки на гребнях волны,
    чтобы было видно, как волна ползёт вдоль тела."""
    fig, ax = plt.subplots(figsize=(6, 6))
    line, = ax.plot([], [], '-o', lw=2, ms=5)
    crest, = ax.plot([], [], 'o', color='tab:red', ms=11)
    txt = ax.text(0.02, 0.96, '', transform=ax.transAxes)
    ax.set_aspect('equal')
    ax.set_xlim(-N * L, N * L)
    ax.set_ylim(-N * L, N * L)
    ax.grid(True)
    ax.set_title('Анимация бегущей волны')

    def update(frame):
        t = frame * DT
        th = angles(t)
        pts = forward_kinematics(th)
        line.set_data(pts[:, 0], pts[:, 1])
        ks = crest_joints(t)                 # гребни волны
        crest.set_data(pts[ks, 0], pts[ks, 1])
        txt.set_text(f't = {t:.2f} c,  гребни в сочленениях {ks}')
        return line, crest, txt

    period_frames = int(round(2 * np.pi / omega / DT))
    ani = FuncAnimation(fig, update, frames=2 * period_frames, interval=30, blit=True)
    ani.save(filename, writer=PillowWriter(fps=30))
    plt.close(fig)
    print(f'GIF сохранён: {filename} ({2 * period_frames} кадров = 2 периода)')


# ---------- П.3. Скорость волны ----------
def crest_times(N_, omega_, delta_, t_end=None, dt=1e-3):
    """Момент первого максимума угла в каждом сочленении (после t = 0)."""
    if t_end is None:
        t_end = (N_ * abs(delta_) + 2 * np.pi) / omega_ + 1
    t = np.arange(0, t_end, dt)
    th = np.array([angles(tt, N=N_, omega=omega_, delta=delta_) for tt in t])
    res = []
    for i in range(N_):
        # локальные максимумы
        idx = np.where((th[1:-1, i] > th[:-2, i]) & (th[1:-1, i] >= th[2:, i]))[0] + 1
        res.append(t[idx[0]])
    return np.array(res)


def measure_speed(N_=N, omega_=omega, delta_=delta):
    """Оценка скорости волны по наклону t_гребня(i).
    Возвращает (численная, теоретическая) скорость в звеньях/с."""
    tc = crest_times(N_, omega_, delta_)
    i = np.arange(N_)
    # гребень приходит в i-е сочленение позже на Δφ/ω (по модулю периода),
    # поэтому «разворачиваем» времена: разность соседей приводим к (-T/2, T/2]
    T = 2 * np.pi / omega_
    d = np.diff(tc)
    d = (d + T / 2) % T - T / 2
    tc_unwrapped = np.concatenate([[tc[0]], tc[0] + np.cumsum(d)])
    slope = np.polyfit(i, tc_unwrapped, 1)[0]      # секунд на звено
    return 1 / slope, omega_ / delta_, i, tc_unwrapped


def task_2_3():
    cases = [(2.0, np.pi / 3), (2.0, np.pi / 6), (4.0, np.pi / 3),
             (2.0, np.pi / 12), (1.0, np.pi / 2)]
    names = ['π/3', 'π/6', 'π/3', 'π/12', 'π/2']

    print(f'{"ω":>4} {"Δφ":>5} | {"v числ.":>8} {"v теор.":>8} звен/с | '
          f'{"λ, звеньев":>10} | {"время пробега тела, c":>21}')
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = ['o-', 's--', 'o-', '^-', 'D-']
    for (w, d), nm, st in zip(cases, names, styles):
        v_num, v_th, i, tc = measure_speed(N, w, d)
        lam = 2 * np.pi / d
        t_body = N * d / w
        print(f'{w:4.1f} {nm:>5} | {v_num:8.3f} {v_th:8.3f}        | '
              f'{lam:10.1f} | {t_body:21.2f}')
        ax.plot(i, tc - tc[0], st, label=f'ω = {w}, Δφ = {nm}: v = {v_num:.2f} звен/с')
    ax.set_xlabel('номер сочленения i')
    ax.set_ylabel('момент прихода гребня (от сочленения 0), c')
    ax.set_title(f'Задание 2.3. Распространение гребня волны вдоль тела (N = {N})')
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig('figures/task2_3_speed.png', dpi=120)

    print('\nВолна всегда бежит от головы (i = 0) к хвосту (i = N-1) при Δφ > 0;')
    print('скорость v = ω/Δφ звеньев/с = ω·L/Δφ ед. длины/с, время пробега N·Δφ/ω.')
    print('При Δφ < 0 направление меняется: волна бежит от хвоста к голове.')
    # проверка обратного направления
    v_neg, _, _, _ = measure_speed(N, 2.0, -np.pi / 3)
    print(f'Проверка: ω = 2, Δφ = -π/3 → v = {v_neg:.3f} звен/с (знак «минус»)')


if __name__ == '__main__':
    print('--- 2.2 ---'); save_gif()
    print('--- 2.3 ---'); task_2_3()
    plt.show()
