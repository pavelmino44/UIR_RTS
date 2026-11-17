"""
Задание 4*. Вариант A — статическая устойчивость гексапода.

Масса распределена по корпусу равномерно, поэтому центр масс (ЦМ) лежит
в геометрическом центре корпуса: (x, y) = (mean(HIP_X), 0) = (0, 0).
Опорный многоугольник — выпуклая оболочка стоп, находящихся в фазе опоры.
Робот статически устойчив, если проекция ЦМ лежит внутри этого многоугольника.

Считаются две величины:
  * запас устойчивости — расстояние от ЦМ до ближайшей стороны опорного
    многоугольника (со знаком: > 0 внутри, < 0 снаружи);
  * расстояние от ЦМ до ближайшей опорной точки (стопы) — как в формулировке задания.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import hexapod_model as hm

os.makedirs('figures', exist_ok=True)
COM = np.array([hm.HIP_X.mean(), 0.0])


# ---------- геометрия ----------
def convex_hull(points):
    """Выпуклая оболочка (алгоритм Эндрю), вершины против часовой стрелки."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def dist_point_segment(p, a, b):
    ab, ap = b - a, p - a
    s = np.clip(np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + s * ab))


def stability_margin(p, hull):
    """Расстояние от точки p до границы многоугольника, > 0 если p внутри."""
    if len(hull) == 0:
        return -np.inf
    if len(hull) == 1:
        return -np.linalg.norm(p - hull[0])
    edges = [(hull[k], hull[(k + 1) % len(hull)]) for k in range(len(hull))]
    d = min(dist_point_segment(p, a, b) for a, b in edges)
    if len(hull) < 3:
        return -d
    # точка внутри выпуклого многоугольника (обход против часовой) —
    # если она слева от всех рёбер
    inside = all((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0
                 for a, b in edges)
    return d if inside else -d


def analyse(duty, scheme, t):
    xs, ys, zs, stance = hm.legs_state(t, duty, scheme)
    feet = np.column_stack([xs, ys])[stance]
    hull = convex_hull(feet)
    margin = stability_margin(COM, hull)
    nearest = np.min(np.linalg.norm(feet - COM, axis=1)) if len(feet) else np.nan
    return margin, nearest, hull, xs, ys, stance


# ---------- расчёт ----------
CASES = [
    (5 / 6, 'original', 'волновая, β = 5/6'),
    (0.5,   'tripod',   'трипод, β = 0.5 (сдвиги {0,2,4}/{1,3,5})'),
    (0.5,   'original', 'β = 0.5 со сдвигами из методички'),
]


def main():
    t = np.arange(0, 2 * hm.T_STEP, 0.002) + 0.001
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    print(f'{"походка":45s} | мин. запас | ср. запас | ЦМ всегда внутри?')
    for duty, scheme, name in CASES:
        res = [analyse(duty, scheme, tt) for tt in t]
        m = np.array([r[0] for r in res])
        near = np.array([r[1] for r in res])
        m_plot = np.where(np.isfinite(m), m, np.nan)
        axes[0].plot(t, m_plot, label=name)
        axes[1].plot(t, near, label=name)
        print(f'{name:45s} | {np.nanmin(m_plot):10.3f} | {np.nanmean(m_plot):9.3f} | '
              f'{"да" if np.all(m > 0) else "НЕТ"}')

    axes[0].axhline(0, color='k', lw=1)
    axes[0].set_ylabel('запас устойчивости, м\n(до ближайшей стороны опорного многоугольника)')
    axes[0].set_title('Задание 4A. Положение проекции ЦМ относительно опорного многоугольника\n'
                      '(> 0 — ЦМ внутри, < 0 — снаружи, робот опрокидывается)')
    axes[1].set_ylabel('расстояние от ЦМ\nдо ближайшей опорной стопы, м')
    axes[1].set_xlabel('t, c')
    for ax in axes:
        ax.grid(True); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig('figures/task4_stability.png', dpi=120)

    # кадры опорного многоугольника для волновой походки
    moments = np.arange(6) / 6 + 1 / 12
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, tt in zip(axes.ravel(), moments):
        margin, near, hull, xs, ys, stance = analyse(5 / 6, 'original', tt)
        poly = np.vstack([hull, hull[:1]])
        ax.fill(poly[:, 0], poly[:, 1], color='tab:blue', alpha=0.15)
        ax.plot(poly[:, 0], poly[:, 1], 'b-', lw=1)
        ax.scatter(xs[stance], ys[stance], c='tab:blue', s=50, label='опора')
        ax.scatter(xs[~stance], ys[~stance], c='tab:red', s=50, label='в воздухе')
        ax.plot(*COM, 'k*', ms=14, label='ЦМ')
        ax.set_title(f't = {tt:.3f} c, запас = {margin:.3f} м', fontsize=10)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.85, 0.85)
        ax.set_aspect('equal'); ax.grid(True)
    axes[0, 0].legend(fontsize=8, loc='center left')
    fig.suptitle('Волновая походка (β = 5/6): опорный многоугольник и проекция ЦМ, вид сверху')
    fig.tight_layout(); fig.savefig('figures/task4_support_polygons.png', dpi=120)


def study_step_len():
    """Дополнительно: влияние длины шага STEP_LEN на запас устойчивости и скорость.
    Скорость корпуса v = STEP_LEN / (β·T): за время опоры β·T стопа проходит STEP_LEN."""
    t = np.arange(0, hm.T_STEP, 0.002) + 0.001
    base = hm.STEP_LEN
    print('\nSTEP_LEN | волновая: мин/ср запас, v | трипод: мин/ср запас, v')
    for L in [0.3, 0.45, 0.6, 0.75, 0.9]:
        hm.STEP_LEN = L
        row = []
        for duty, scheme in [(5 / 6, 'original'), (0.5, 'tripod')]:
            m = np.array([analyse(duty, scheme, tt)[0] for tt in t])
            row.append(f'{m.min():6.3f} / {m.mean():5.3f}, v = {L / (duty * hm.T_STEP):4.2f}')
        print(f'  {L:4.2f}   | {row[0]} | {row[1]}')
    hm.STEP_LEN = base


if __name__ == '__main__':
    main()
    study_step_len()
    plt.show()
