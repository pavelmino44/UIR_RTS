"""
Задание 1 (пункты 2–4). Исследование кинематики змееподобного робота.
Использует функции angles() и forward_kinematics() из snake_wave.py без изменений.
Графики сохраняются в папку figures/ и показываются на экране.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from snake_wave import angles, forward_kinematics, A, omega, delta, L

os.makedirs('figures', exist_ok=True)
TIMES = [0.0, 0.4, 0.8, 1.2, 1.6]


def draw_snakes(ax, N, delta_=delta, times=TIMES, shift=None, title=None):
    """Рисует змею длины N в несколько моментов времени.
    Каждый снимок сдвигается по оси X, чтобы формы не накладывались."""
    if shift is None:
        shift = 0.6 * N * L
    for k, t in enumerate(times):
        pts = forward_kinematics(angles(t, N=N, delta=delta_))
        ax.plot(pts[:, 0] + k * shift, pts[:, 1], '-o', ms=3, lw=1.5,
                label=f't = {t:.1f} c')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title or f'N = {N}')
    ax.set_xlabel('x (со сдвигом для каждого t)')
    ax.set_ylabel('y')


# ---------- П.2. Форма змеи при N = 6, 12, 20 ----------
def task_1_2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, n in zip(axes, [6, 12, 20]):
        draw_snakes(ax, n)
    axes[0].legend(fontsize=8)
    fig.suptitle('Задание 1.2. Влияние числа звеньев N на форму змеи '
                 f'(A = {A}, Δφ = π/3)')
    fig.tight_layout()
    fig.savefig('figures/task1_2_N.png', dpi=120)

    # длина волны в звеньях: на неё приходится полный период синуса
    lam = 2 * np.pi / delta
    for n in [6, 12, 20]:
        print(f'N = {n:2d}: на теле укладывается {n / lam:.2f} длины волны '
              f'(λ = {lam:.0f} звеньев)')


# ---------- П.3. angles(t)[0] и angles(t)[5] ----------
def task_1_3():
    t = np.linspace(0, 10, 5001)
    th = np.array([angles(tt) for tt in t])
    th0, th5 = th[:, 0], th[:, 5]

    lag_theory = 5 * delta / omega
    # численная оценка запаздывания: сдвиг, при котором th0(t - τ) ≈ th5(t)
    dt = t[1] - t[0]
    T = 2 * np.pi / omega
    taus = np.arange(0, T, dt)
    err = [np.mean((np.interp(t[t > T] - tau, t, th0) - th5[t > T]) ** 2)
           for tau in taus]
    lag_num = taus[int(np.argmin(err))]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, th0, label='angles(t)[0] — сочленение 0')
    ax.plot(t, th5, label='angles(t)[5] — сочленение 5')
    # показать запаздывание на одном максимуме
    t_max0 = (np.pi / 2) / omega + T          # максимум θ0
    t_max5 = t_max0 + lag_theory              # соответствующий максимум θ5
    ax.annotate('', xy=(t_max5, A * 1.05), xytext=(t_max0, A * 1.05),
                arrowprops=dict(arrowstyle='<->', color='k'))
    ax.text((t_max0 + t_max5) / 2, A * 1.12,
            f'5·Δφ/ω = {lag_theory:.3f} c', ha='center')
    ax.set_ylim(-A * 1.6, A * 1.35)
    ax.set_xlabel('t, c')
    ax.set_ylabel('угол, рад')
    ax.set_title('Задание 1.3. Углы в сочленениях 0 и 5')
    ax.grid(True)
    ax.legend(loc='lower center', ncol=2)
    fig.tight_layout()
    fig.savefig('figures/task1_3_angles.png', dpi=120)

    print(f'Теоретическое запаздывание 5·Δφ/ω = {lag_theory:.3f} c')
    print(f'Численная оценка запаздывания     = {lag_num:.3f} c')


# ---------- П.4. Уменьшение delta до π/12 ----------
def task_1_4():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    draw_snakes(axes[0], 12, delta_=np.pi / 3, title='Δφ = π/3')
    draw_snakes(axes[1], 12, delta_=np.pi / 12, shift=10, title='Δφ = π/12')
    axes[0].legend(fontsize=8)
    fig.suptitle('Задание 1.4. Влияние фазового сдвига на форму (N = 12)')
    fig.tight_layout()
    fig.savefig('figures/task1_4_delta.png', dpi=120)

    # максимумы за период T = 2π/ω
    t_period = np.linspace(0, 2 * np.pi / omega, 400)
    for d, name in [(np.pi / 3, 'π/3'), (np.pi / 12, 'π/12')]:
        jump = bend = 0.0
        for t in t_period:
            th = angles(t, N=12, delta=d)
            jump = max(jump, np.max(np.abs(np.diff(th))))    # скачок угла между соседями
            bend = max(bend, np.max(np.abs(np.cumsum(th))))  # накопленный поворот корпуса
        print(f'Δφ = {name:5s}: макс. разница углов соседей = {jump:.3f} рад, '
              f'макс. накопленный поворот = {np.degrees(bend):.0f}°')


if __name__ == '__main__':
    print('--- 1.2 ---'); task_1_2()
    print('--- 1.3 ---'); task_1_3()
    print('--- 1.4 ---'); task_1_4()
    plt.show()
