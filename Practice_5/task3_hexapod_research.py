"""
Задание 3. Волновая походка шестиногого робота.
  3.1 — проверка: при DUTY = 5/6 в воздухе ровно одна нога;
  3.2 — DUTY = 0.5: трипод, скриншоты;
  3.3 — зависимость числа ног в воздухе n_swing от DUTY в диапазоне [0.5, 0.9].
Модель — hexapod_model.py (те же формулы, что в hexapod_wave.py).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import hexapod_model as hm

os.makedirs('figures', exist_ok=True)
# моменты времени сдвинуты на полшага, чтобы не попадать точно на границу опора/перенос
T_GRID = np.arange(0, 3 * hm.T_STEP, 0.001) + 0.0005


def swing_series(duty, scheme='original'):
    return np.array([hm.n_swing(t, duty, scheme) for t in T_GRID])


def draw_robot(ax_side, ax_top, t, duty, scheme, title):
    """Кадр робота: вид сбоку (как в исходной анимации) и вид сверху."""
    xs, ys, zs, stance = hm.legs_state(t, duty, scheme)
    # --- вид сбоку ---
    ax_side.plot([hm.HIP_X.min(), hm.HIP_X.max()], [0, 0], 'k-', lw=4)
    for i in range(hm.N_LEGS):
        c = 'tab:blue' if stance[i] else 'tab:red'
        ax_side.plot([hm.HIP_X[i], xs[i]], [0, zs[i]], '-', color=c, lw=2)
        ax_side.plot(xs[i], zs[i], 'o', color=c, ms=8)
    ax_side.set_xlim(-1.5, 1.5); ax_side.set_ylim(-0.3, 0.5)
    ax_side.set_aspect('equal'); ax_side.grid(True)
    ax_side.set_title(f'{title}\nвид сбоку, t = {t:.2f} c')
    # --- вид сверху ---
    ax_top.add_patch(plt.Rectangle((hm.HIP_X.min() - 0.15, -hm.BODY_W),
                                   np.ptp(hm.HIP_X) + 0.3, 2 * hm.BODY_W,
                                   color='lightgray', zorder=0))
    for i in range(hm.N_LEGS):
        c = 'tab:blue' if stance[i] else 'tab:red'
        hy = hm.HIP_SIDE[i] * hm.BODY_W
        ax_top.plot([hm.HIP_X[i], xs[i]], [hy, ys[i]], '-', color=c, lw=2)
        ax_top.plot(xs[i], ys[i], 'o', color=c, ms=9)
        ax_top.text(xs[i], ys[i] + 0.13 * hm.HIP_SIDE[i], str(i),
                    ha='center', va='center', fontsize=9)
    ax_top.set_xlim(-1.5, 1.5); ax_top.set_ylim(-0.9, 0.9)
    ax_top.set_aspect('equal'); ax_top.grid(True)
    ax_top.set_title(f'вид сверху: в воздухе {int(np.sum(~stance))} '
                     f'(красные), в опоре {int(np.sum(stance))} (синие)')
    ax_top.set_xlabel('x (вперёд →)')


# ---------- 3.1 ----------
def task_3_1():
    n = swing_series(5 / 6)
    print(f'DUTY = 5/6: число ног в воздухе принимает значения {sorted(set(n.tolist()))}')
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.step(T_GRID, n, where='post')
    ax.set_ylim(-0.2, 5.5); ax.grid(True)
    ax.set_xlabel('t, c'); ax.set_ylabel('ног в воздухе')
    ax.set_title('Задание 3.1. Волновая походка (DUTY = 5/6): всегда ровно одна нога в воздухе')
    fig.tight_layout(); fig.savefig('figures/task3_1_wave_nswing.png', dpi=120)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6))
    draw_robot(axes[0], axes[1], 0.30, 5 / 6, 'original', 'DUTY = 5/6 (волновая)')
    fig.tight_layout(); fig.savefig('figures/task3_1_wave_screenshot.png', dpi=120)


# ---------- 3.2 ----------
def task_3_2():
    n_orig = swing_series(0.5, 'original')
    n_trip = swing_series(0.5, 'tripod')
    print(f'DUTY = 0.5, сдвиги из методички (k·DUTY/5): '
          f'ног в воздухе {sorted(set(n_orig.tolist()))}, в среднем {n_orig.mean():.2f}')
    print(f'DUTY = 0.5, сдвиги триподной схемы {{0,2,4}}/{{1,3,5}}: '
          f'ног в воздухе {sorted(set(n_trip.tolist()))}, в среднем {n_trip.mean():.2f}')

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.step(T_GRID, n_orig, where='post', label='сдвиги из методички: k·DUTY/(N-1)')
    ax.step(T_GRID, n_trip, where='post', lw=2.5, label='трипод: ноги {0,2,4} и {1,3,5} в противофазе')
    ax.set_ylim(-0.2, 6); ax.grid(True); ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('t, c'); ax.set_ylabel('ног в воздухе')
    ax.set_title('Задание 3.2. DUTY = 0.5')
    fig.tight_layout(); fig.savefig('figures/task3_2_tripod_nswing.png', dpi=120)

    # «скриншоты»
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    draw_robot(axes[0, 0], axes[1, 0], 0.25, 0.5, 'tripod',
               'DUTY = 0.5, триподная схема сдвигов')
    draw_robot(axes[0, 1], axes[1, 1], 0.05, 0.5, 'original',
               'DUTY = 0.5, сдвиги из методички')
    fig.tight_layout(); fig.savefig('figures/task3_2_tripod_screenshot.png', dpi=120)


# ---------- 3.3 ----------
def task_3_3():
    duties = np.round(np.arange(0.50, 0.9001, 0.01), 3)
    mean_, min_, max_ = [], [], []
    for d in duties:
        n = swing_series(d)
        mean_.append(n.mean()); min_.append(n.min()); max_.append(n.max())
    mean_, min_, max_ = map(np.array, (mean_, min_, max_))
    theory = hm.N_LEGS * (1 - duties)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(duties, min_, max_, step='mid', alpha=0.2,
                    label='диапазон мгновенных значений (min…max)')
    ax.plot(duties, mean_, 'o', ms=4, label='среднее за период (численно)')
    ax.plot(duties, theory, 'k--', label='теория: 6·(1 − β)')
    ax.axvline(5 / 6, color='gray', ls=':'); ax.text(5 / 6 + 0.005, 4.5, 'β = 5/6')
    ax.set_xlabel('DUTY (β)'); ax.set_ylabel('n_swing — ног в воздухе')
    ax.set_title('Задание 3.3. Число ног в воздухе в зависимости от DUTY')
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig('figures/task3_3_nswing_duty.png', dpi=120)

    print(' DUTY | среднее | теория 6(1-β) | min | max')
    for d, m, th, a, b in zip(duties, mean_, theory, min_, max_):
        if abs(d * 100 % 5) < 1e-6 or abs(d - 0.83) < 1e-6:
            print(f' {d:.2f} | {m:7.3f} | {th:13.3f} | {a:3d} | {b:3d}')


if __name__ == '__main__':
    print('--- 3.1 ---'); task_3_1()
    print('--- 3.2 ---'); task_3_2()
    print('--- 3.3 ---'); task_3_3()
    plt.show()
