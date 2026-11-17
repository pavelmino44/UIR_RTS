import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from snake_wave import angles, forward_kinematics  # используем функции из задания 1
# N и L берутся из snake_wave.
# В методичке этот импорт стоит ниже, после set_xlim(-N * L, ...),
# из-за чего скрипт падает с NameError. Перенесён наверх.
from snake_wave import N, L

fig, ax = plt.subplots(figsize=(8, 8))
line, = ax.plot([], [], '-o', lw=2, ms=5)
ax.set_aspect('equal')
ax.set_xlim(-N * L, N * L)
ax.set_ylim(-N * L, N * L)
ax.grid(True)
ax.set_title('Анимация бегущей волны')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    t = frame * 0.05
    pts = forward_kinematics(angles(t))
    line.set_data(pts[:, 0], pts[:, 1])
    return line,

ani = FuncAnimation(fig, update, frames=400, init_func=init, blit=True, interval=30)
plt.show()
# для сохранения: ani.save('snake_wave.gif', writer='pillow', fps=30)
