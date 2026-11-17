import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Параметры ----------
N_LEGS      = 6
DUTY        = 5/6           # β — доля времени в опоре (5/6 — волновая, 0.5 — трипод)
STEP_LEN    = 0.6           # длина шага
SWING_H     = 0.25          # высота подъёма ноги
T_STEP      = 1.0           # период шага
HIP_X       = np.array([0.6, 0.0, -0.6, 0.6, 0.0, -0.6])  # x позиции тазобедренных шарниров
HIP_SIDE    = np.array([+1, +1, +1, -1, -1, -1])           # левая/правая сторона

# Фазовый сдвиг: волновая походка — по одной ноге в воздухе
# Порядок «волны» по ногам: 0 → 3 → 1 → 4 → 2 → 5
PHASE_ORDER = np.array([0, 2, 4, 1, 3, 5])
PHASE_OFFSET = np.array([0, 1, 2, 3, 4, 5]) * DUTY / (N_LEGS - 1)

# ---------- Расчёт стопы для одной ноги ----------
def foot_trajectory(phase, side):
    """
    phase in [0, 1) — фаза ноги,
    возвращает (x, z) — позицию стопы в системе координат ноги.
    """
    if phase < DUTY:                        # опора: стопа идёт назад
        s = phase / DUTY
        x = STEP_LEN * (0.5 - s)
        z = 0.0
    else:                                   # перенос: стопа идёт вперёд по дуге
        s = (phase - DUTY) / (1 - DUTY)
        x = STEP_LEN * (-0.5 + s)
        z = SWING_H * np.sin(np.pi * s)
    return x, z

def legs_state(t):
    """Вернуть массивы позиций всех стоп в глобальных координатах."""
    xs, zs = np.zeros(N_LEGS), np.zeros(N_LEGS)
    for i in range(N_LEGS):
        # фаза ноги i с учётом её «номера в волне»
        phase = (t / T_STEP + PHASE_OFFSET[np.where(PHASE_ORDER == i)[0][0]]) % 1.0
        fx, fz = foot_trajectory(phase, HIP_SIDE[i])
        xs[i] = HIP_X[i] + fx
        zs[i] = fz
    return xs, zs



# ---------- Анимация ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 1.0)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title('Волновая походка гексапода (вид сбоку)')

body_line, = ax.plot([], [], 'k-', lw=4)
leg_lines = [ax.plot([], [], '-', lw=2)[0] for _ in range(N_LEGS)]
foot_dots = [ax.plot([], [], 'o', ms=8)[0] for _ in range(N_LEGS)]

def update(frame):
    t = frame * 0.02
    xs, zs = legs_state(t)
    body_line.set_data([HIP_X.min(), HIP_X.max()], [0, 0])
    for i in range(N_LEGS):
        in_air = zs[i] > 1e-3
        color = 'tab:red' if in_air else 'tab:blue'
        leg_lines[i].set_data([HIP_X[i], xs[i]], [0, zs[i]])
        leg_lines[i].set_color(color)
        foot_dots[i].set_data([xs[i]], [zs[i]])
        foot_dots[i].set_color(color)
    return [body_line, *leg_lines, *foot_dots]

ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)
plt.show()
# ani.save('hexapod_wave.gif', writer='pillow', fps=30)
