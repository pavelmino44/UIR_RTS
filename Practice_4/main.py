from __future__ import annotations

import math
import threading
import queue
from dataclasses import dataclass, field, replace

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch, Polygon as PolygonPatch
from matplotlib.path import Path as MplPath

import tkinter as tk
from tkinter import ttk

EPS = 1e-9


# ============================================================================
#  Вспомогательные функции
# ============================================================================
def distance(p1, p2) -> float:
    """Евклидово расстояние (п.5)."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if (n <= max_norm or n < EPS) else v * (max_norm / n)


def rotate(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros(2)


# ============================================================================
#  Конфигурация 
# ============================================================================
@dataclass
class Config:
    width: float = 10.0
    height: float = 10.0
    step_size: float = 0.1
    max_steps: int = 500
    robot_radius: float = 0.15
    clearance_margin: float = 0.02
    goal_tol: float = 0.15
    k_att: float = 0.5
    k_rep: float = 2.0
    d_safe: float = 2.0
    d_min: float = 0.05
    potential: str = "quadratic"
    d_switch: float = 1.0
    max_force: float = 20.0
    normalize: bool = False
    smoothing: float = 0.0
    adaptive_rep: bool = False
    adaptive_alpha: float = 2.0
    dynamic: bool = False
    prediction_horizon: float = 0.0
    gnron_fix: bool = False
    gnron_power: int = 2
    escape: str = "tangential"
    escape_gain: float = 1.5
    escape_steps: int = 25
    stagnation_window: int = 20
    stagnation_tol: float = 0.05
    chain_spacing: float = 0.8      # дистанция между роботами в цепи (по пути)
    inter_robot_safe: float = 0.4   # зона отталкивания между роботами (< spacing - 2r)
    follower_tol: float = 0.05      # ведомый встал на своё место в строю
    formation_tol: float = 0.4      # макс. отклонение ведомого, если он остановился
    settle_tol: float = 0.003       # шаг, меньше которого робот считается остановившимся
    # «Гонка за лидером»
    pursuit_spacing: float = 0.8    # дистанция, на которой ведомые держатся от лидера
    # «Свободный поиск»
    sensor_range: float = 2.0       # радиус обнаружения цели датчиком робота
    search_max_steps: int = 1500    # лимит тактов для поиска (цель заранее неизвестна)
    waypoint_tol: float = 0.3       # точка обследования считается достигнутой
    waypoint_timeout: int = 150     # макс. число тактов на одну точку обследования
    waypoint_clearance: float = 0.3 # мин. зазор от точки обследования до препятствий
    slot_radius: float = 0.8        # радиус сбора группы вокруг найденной цели
    seed: int = 42


# ============================================================================
#  Препятствия 
# ============================================================================
@dataclass(kw_only=True)
class Obstacle:
    k_rep_scale: float = 1.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    d_safe_override: float | None = None

    def safe_zone(self, cfg: Config) -> float:
        return cfg.d_safe if self.d_safe_override is None else self.d_safe_override

    def surface_distance(self, p: np.ndarray):
        raise NotImplementedError

    def distance_field(self, X, Y):
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError

    def centroid(self) -> np.ndarray:
        raise NotImplementedError

    def _translate(self, dp: np.ndarray) -> None:
        raise NotImplementedError

    def draw(self, ax, **kw) -> None:
        raise NotImplementedError

    def predicted(self, tau: float) -> "Obstacle":
        if tau <= 0.0 or not np.any(self.velocity):
            return self
        return self._shifted(self.velocity * tau)

    def _shifted(self, dp: np.ndarray) -> "Obstacle":
        raise NotImplementedError

    def advance(self, dt: float, cfg: Config) -> None:
        """Перемещение с отражением от границ (вариант 4)."""
        dp = self.velocity * dt
        if not np.any(dp):
            return
        self._translate(dp)
        lo, hi = self.bounds()
        corr = np.zeros(2)
        for i, lim in enumerate((cfg.width, cfg.height)):
            if lo[i] < 0.0:
                corr[i] = -lo[i]
                self.velocity[i] *= -1.0
            elif hi[i] > lim:
                corr[i] = lim - hi[i]
                self.velocity[i] *= -1.0
        if np.any(corr):
            self._translate(corr)


@dataclass
class CircleObstacle(Obstacle):
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()

    def surface_distance(self, p):
        v = np.asarray(p, dtype=float) - self.center
        d = float(np.linalg.norm(v))
        if d < EPS:
            return -self.radius, np.array([1.0, 0.0])
        return d - self.radius, v / d

    def distance_field(self, X, Y):
        return np.hypot(X - self.center[0], Y - self.center[1]) - self.radius

    def bounds(self):
        r = np.array([self.radius, self.radius])
        return self.center - r, self.center + r

    def centroid(self):
        return self.center.copy()

    def _translate(self, dp):
        self.center = self.center + dp

    def _shifted(self, dp):
        return replace(self, center=self.center + dp)

    def draw(self, ax, **kw):
        ax.add_patch(CirclePatch(self.center, self.radius, **kw))


@dataclass
class PolygonObstacle(Obstacle):
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((3, 2)))

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).copy()
        self.velocity = np.asarray(self.velocity, dtype=float).copy()
        self._path = MplPath(self.vertices, closed=True)

    def _edges(self):
        a = self.vertices
        b = np.roll(a, -1, axis=0)
        return a, b - a

    def surface_distance(self, p):
        p = np.asarray(p, dtype=float)
        a, ab = self._edges()
        den = np.maximum(np.einsum("ij,ij->i", ab, ab), EPS)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / den, 0.0, 1.0)
        proj = a + t[:, None] * ab
        diff = p - proj
        dist = np.linalg.norm(diff, axis=1)
        i = int(np.argmin(dist))
        inside = bool(self._path.contains_point(p))
        d = float(dist[i])
        if d < 1e-9:
            e = ab[i] / max(float(np.linalg.norm(ab[i])), EPS)
            n = np.array([e[1], -e[0]])
            return 0.0, (-n if inside else n)
        g = diff[i] / d
        return (-d if inside else d), (-g if inside else g)

    def distance_field(self, X, Y):
        P = np.stack([X, Y], axis=-1)
        best = np.full(X.shape, np.inf)
        a, ab = self._edges()
        den = np.maximum(np.einsum("ij,ij->i", ab, ab), EPS)
        for ai, abi, dni in zip(a, ab, den):
            w = P - ai
            t = np.clip((w @ abi) / dni, 0.0, 1.0)
            proj = ai + t[..., None] * abi
            best = np.minimum(best, np.linalg.norm(P - proj, axis=-1))
        inside = self._path.contains_points(P.reshape(-1, 2)).reshape(X.shape)
        return np.where(inside, -best, best)

    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def centroid(self):
        return self.vertices.mean(axis=0)

    def _translate(self, dp):
        self.vertices = self.vertices + dp
        self._path = MplPath(self.vertices, closed=True)

    def _shifted(self, dp):
        return replace(self, vertices=self.vertices + dp)

    def draw(self, ax, **kw):
        ax.add_patch(PolygonPatch(self.vertices, closed=True, **kw))


def rectangle(cx, cy, w, h, angle_deg=0.0, **kw) -> PolygonObstacle:
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return PolygonObstacle(vertices=local @ R.T + np.array([cx, cy]), **kw)


def regular_polygon(cx, cy, r, n=5, phase=0.0, **kw) -> PolygonObstacle:
    ang = phase + np.arange(n) * 2 * math.pi / n
    v = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
    return PolygonObstacle(vertices=v, **kw)


# ============================================================================
#  Силы 
# ============================================================================
def attractive_force(current_pos, goal_pos, cfg: Config) -> np.ndarray:
    """Сила притяжения (п.6 методички). Три вида потенциала (п.7)."""
    diff = np.asarray(goal_pos, float) - np.asarray(current_pos, float)
    d = float(np.linalg.norm(diff))
    if d < EPS:
        return np.zeros(2)
    if cfg.potential == "quadratic":
        return cfg.k_att * diff
    if cfg.potential == "linear":
        return cfg.k_att * diff / d
    if cfg.potential == "hybrid":
        if d <= cfg.d_switch:
            return cfg.k_att * diff
        return cfg.k_att * cfg.d_switch * diff / d
    raise ValueError(f"Неизвестный потенциал: {cfg.potential}")


def _repulsion(d_clear, grad, k_rep, d0, goal_vec, cfg: Config) -> np.ndarray:
    """Ядро Хатиба + коррекция GNRON."""
    if d_clear > d0:
        return np.zeros(2)
    d = max(d_clear, cfg.d_min)
    inv = max(1.0 / d - 1.0 / d0, 0.0)
    mag = k_rep * inv / (d * d)
    if not cfg.gnron_fix:
        return mag * grad
    dg = float(np.linalg.norm(goal_vec))
    n = cfg.gnron_power
    F = mag * (dg ** n) * grad
    if dg > EPS:
        F += 0.5 * n * k_rep * (inv ** 2) * (dg ** (n - 1)) * (goal_vec / dg)
    return F


def repulsive_force(current_pos, obstacle: Obstacle, cfg: Config, goal_pos) -> np.ndarray:
    """Сила отталкивания (п.7 методички)."""
    d_surf, grad = obstacle.surface_distance(current_pos)
    d_clear = d_surf - cfg.robot_radius
    d0 = obstacle.safe_zone(cfg)
    k = cfg.k_rep * obstacle.k_rep_scale
    if cfg.adaptive_rep:
        k *= 1.0 + cfg.adaptive_alpha * max(0.0, 1.0 - d_clear / d0)
    return _repulsion(d_clear, grad, k, d0, np.asarray(goal_pos) - current_pos, cfg)


def total_force(pos, goal, obstacles, cfg: Config):
    """Результирующая сила (п.8 методички)."""
    F_att = attractive_force(pos, goal, cfg)
    F_rep = np.zeros(2)
    for obs in obstacles:
        F_rep += repulsive_force(pos, obs, cfg, goal)
    F_rep = clip_norm(F_rep, cfg.max_force)
    return F_att + F_rep, F_att, F_rep


# ============================================================================
#  Состояние робота и шаг
# ============================================================================
@dataclass
class RobotState:
    name: str
    pos: np.ndarray
    path: list = field(default_factory=list)
    history: list = field(default_factory=list)
    dir_ema: np.ndarray = field(default_factory=lambda: np.zeros(2))
    escape_left: int = 0
    escape_sign: float = 1.0
    escapes: int = 0
    done: bool = False              # робот прибыл и больше не двигается
    # для «свободного поиска»
    waypoint: np.ndarray | None = None   # текущая точка обследования
    waypoint_tick: int = 0               # такт, когда точка была назначена
    waypoint_escapes: int = 0            # счётчик выходов из минимума на момент назначения
    visited: list = field(default_factory=list)  # пройденные точки обследования

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float).copy()
        self.path = [self.pos.copy()]


def min_clearance(pos, obstacles, cfg: Config) -> float:
    if not obstacles:
        return math.inf
    return min(o.surface_distance(pos)[0] - cfg.robot_radius for o in obstacles)


def in_bounds(p, cfg: Config) -> bool:
    return 0.0 <= p[0] <= cfg.width and 0.0 <= p[1] <= cfg.height


def _line_search(pos, delta, ok, iters=14) -> float:
    if ok(pos + delta):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(pos + mid * delta):
            lo = mid
        else:
            hi = mid
    return lo


def safe_move(pos, delta, obstacles, cfg: Config):
    if float(np.linalg.norm(delta)) < EPS:
        return pos, False
    c0 = min_clearance(pos, obstacles, cfg)
    thr = min(cfg.clearance_margin, c0 - 1e-6)

    def ok(p):
        return in_bounds(p, cfg) and min_clearance(p, obstacles, cfg) >= thr

    a = _line_search(pos, delta, ok)
    if a >= 0.2:
        return pos + a * delta, False
    _, grad = min(((o.surface_distance(pos)[0], o.surface_distance(pos)[1]) for o in obstacles),
                  key=lambda t: t[0]) if obstacles else (math.inf, np.zeros(2))
    slide = delta - min(0.0, float(delta @ grad)) * grad
    if float(np.linalg.norm(slide)) > EPS:
        a2 = _line_search(pos, slide, ok)
        if a2 > 1e-3:
            return pos + a2 * slide, False
    return pos, True


def compute_step(state: RobotState, goal, obstacles, cfg: Config, rng):
    """Один шаг (п.8)."""
    F, F_att, F_rep = total_force(state.pos, goal, obstacles, cfg)

    # Детектор локального минимума (п.6)
    f_norm = float(np.linalg.norm(F))
    if distance(state.pos, goal) > cfg.goal_tol:
        if f_norm < 1e-3 or (len(state.history) >= cfg.stagnation_window and
                             distance(state.pos, state.history[0]) < cfg.stagnation_tol):
            if cfg.escape != "none":
                _, ng = min(((o.surface_distance(state.pos)[0], o.surface_distance(state.pos)[1])
                             for o in obstacles), key=lambda t: t[0]) if obstacles else (math.inf, np.zeros(2))
                if state.escape_left <= 0:
                    dp = float(rotate(ng, math.pi / 2) @ unit(np.asarray(goal) - state.pos))
                    dm = float(rotate(ng, -math.pi / 2) @ unit(np.asarray(goal) - state.pos))
                    state.escape_sign = 1.0 if dp >= dm else -1.0
                    state.escape_left = cfg.escape_steps
                    state.escapes += 1
                state.escape_left -= 1
                amp = cfg.escape_gain * max(float(np.linalg.norm(F_att)), 0.1)
                F = F + amp * rotate(ng, state.escape_sign * math.pi / 2)

    n = float(np.linalg.norm(F))
    if n < EPS:
        return np.zeros(2), F
    u = F / n
    if cfg.smoothing > 0.0:
        state.dir_ema = cfg.smoothing * state.dir_ema + (1.0 - cfg.smoothing) * u
        m = float(np.linalg.norm(state.dir_ema))
        if m > EPS:
            u = state.dir_ema / m
    if cfg.normalize:
        delta = cfg.step_size * u
    else:
        delta = cfg.step_size * F / max(1.0, n)
    delta = clip_norm(delta, cfg.step_size)
    dg = distance(state.pos, goal)
    if float(np.linalg.norm(delta)) > dg:
        delta = unit(delta) * dg
    return delta, F


def _advance_state(state, delta, goal, real_obstacles, cfg):
    state.history.append(state.pos.copy())
    while len(state.history) > cfg.stagnation_window:
        state.history.pop(0)
    new_pos, _ = safe_move(state.pos, delta, real_obstacles, cfg)
    state.pos = new_pos
    state.path.append(new_pos.copy())


# ============================================================================
#  Распределённая система «цепь» 
# ============================================================================
def breadcrumb(path: list, spacing: float):
    """Точка на пути предшественника позади него, на расстоянии spacing по прямой.

    Идём по пути назад и берём первую точку, удалённую от текущего положения
    предшественника на spacing. Расстояние именно по прямой: если путь петляет
    (робот топчется на месте), точка всё равно не окажется вплотную к нему.
    Если предшественник ещё не отъехал на spacing — возвращает None.
    """
    head = np.asarray(path[-1], float)
    for i in range(len(path) - 2, -1, -1):
        b = np.asarray(path[i], float)
        if distance(b, head) >= spacing:
            a = np.asarray(path[i + 1], float)
            v, w = b - a, a - head
            vv, wv, ww = float(v @ v), float(w @ v), float(w @ w)
            if vv < EPS:
                return b.copy()
            t = (-wv + math.sqrt(max(wv * wv - vv * (ww - spacing ** 2), 0.0))) / vv
            return a + min(max(t, 0.0), 1.0) * v
    return None


def convoy_starts(start, goal, cfg: Config, n, spacing):
    """Размещение строя: лидер в старте, ведомые в линию за ним.

    Сначала пробуем поставить ведомых позади лидера (против направления на цель).
    Если строй не помещается в рабочую область (старт в углу), пробуем
    перпендикулярные направления и оси координат.
    """
    s = np.asarray(start, float)
    u = unit(np.asarray(goal) - s)
    candidates = [-u, rotate(u, math.pi / 2), rotate(u, -math.pi / 2),
                  np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                  np.array([-1.0, 0.0]), np.array([0.0, -1.0])]
    for d in candidates:
        pts = [s + i * spacing * d for i in range(n)]
        if all(in_bounds(p, cfg) for p in pts):
            return pts
    return [s.copy() for _ in range(n)]


# ============================================================================
#  Распределённая система «гонка за лидером»
# ============================================================================
def pursuit_point(leader_pos, follower_pos, spacing: float) -> np.ndarray:
    """Точка преследования: на расстоянии spacing от лидера со стороны ведомого.

    Ведомый движется не по пути лидера (как в цепи), а прямо к его текущему
    положению и останавливается на дистанции spacing. Направление подхода
    не фиксировано, поэтому ведомые сами распределяются вокруг лидера.
    """
    lp = np.asarray(leader_pos, float)
    u = unit(np.asarray(follower_pos, float) - lp)
    if not np.any(u):
        u = np.array([-1.0, 0.0])
    return lp + spacing * u


# ============================================================================
#  Распределённая система «свободный поиск»
# ============================================================================
def search_strip(i: int, n: int, cfg: Config):
    """Полоса рабочей области (по оси x), которую обследует робот i из n."""
    w = cfg.width / n
    return i * w, (i + 1) * w


def new_waypoint(i: int, n: int, robot: RobotState, obstacles, cfg: Config, rng):
    """Случайная точка обследования в своей полосе.

    Точка выбирается равномерно в полосе робота, не ближе waypoint_clearance
    к препятствиям и не ближе sensor_range к уже обследованным точкам
    (чтобы робот не возвращался в осмотренные места).
    """
    x0, x1 = search_strip(i, n, cfg)
    m = cfg.robot_radius + 0.15
    best, best_score = None, -1.0
    for _ in range(60):
        p = np.array([rng.uniform(x0 + m, x1 - m), rng.uniform(m, cfg.height - m)])
        if min_clearance(p, obstacles, cfg) < cfg.waypoint_clearance:
            continue
        pts = robot.visited + [robot.pos]
        score = min(distance(p, q) for q in pts)
        if score >= cfg.sensor_range:
            return p
        if score > best_score:
            best, best_score = p, score
    return best if best is not None else robot.pos.copy()


def gather_slots(goal, n: int, obstacles, cfg: Config):
    """Места сбора группы вокруг найденной цели.

    Первое место — сама цель (для робота, который её обнаружил), остальные —
    на окружности slot_radius вокруг цели, со стороны центра рабочей области.
    Места внутри препятствий или за границами области пропускаются.
    """
    g = np.asarray(goal, float)
    center = np.array([cfg.width / 2, cfg.height / 2])
    base = math.atan2(*(center - g)[::-1]) if distance(center, g) > EPS else 0.0
    slots = [g.copy()]
    for off_deg in (25, -25, 70, -70, 115, -115, 160, -160):
        if len(slots) >= n:
            break
        a = base + math.radians(off_deg)
        p = g + cfg.slot_radius * np.array([math.cos(a), math.sin(a)])
        m = cfg.robot_radius
        if not (m <= p[0] <= cfg.width - m and m <= p[1] <= cfg.height - m):
            continue
        if min_clearance(p, obstacles, cfg) < cfg.clearance_margin:
            continue
        if any(distance(p, q) < 2.5 * cfg.robot_radius for q in slots):
            continue
        slots.append(p)
    while len(slots) < n:
        slots.append(g.copy())
    return slots


# ============================================================================
def build_scenario(name: str, cfg: Config):
    if name == "base":
        # Препятствия движутся ПЕРПЕНДИКУЛЯРНО диагонали старта-цели
        # и гарантированно пересекают путь робота 2-3 раза
        return (np.array([0.0, 0.0]), np.array([9.0, 9.0]),
                [CircleObstacle(center=np.array([2.5, 4.5]), radius=1.0,
                                velocity=np.array([0.12, 0.00])),   # движется по X
                 CircleObstacle(center=np.array([5.5, 2.5]), radius=1.2,
                                velocity=np.array([0.00, 0.12]))])  # движется по Y

    if name == "trap":
        # Статичная ловушка — препятствие на диагонали
        return (np.array([0.0, 0.0]), np.array([9.0, 9.0]),
                [CircleObstacle(center=np.array([4.5, 4.5]), radius=1.5)])

    if name == "wall":
        # Статичная стенка — локальный минимум
        return (np.array([0.0, 0.0]), np.array([9.0, 9.0]),
                [rectangle(4.5, 4.5, 5.0, 0.6, angle_deg=135.0)])

    if name == "mixed":

        return (np.array([0.0, 0.0]), np.array([9.0, 9.0]),
                [CircleObstacle(center=np.array([2.5, 4.0]), radius=1.0,
                                velocity=np.array([0.10, 0.00])),
                 rectangle(6.0, 6.0, 2.2, 1.2, angle_deg=30.0,
                           velocity=np.array([-0.05, -0.05])),
                 regular_polygon(7.5, 3.0, 1.0, n=5, phase=0.3),   # статичный
                 CircleObstacle(center=np.array([5.0, 2.0]), radius=0.9,
                                velocity=np.array([0.00, 0.11]))])

    if name == "gnron":
        # GNRON: цель ВНЕ препятствия, но близко (классический случай)
        # Препятствие в (7.5, 7.5), r=1.2; цель (8.8, 8.8)
        # d_surface = 1.838 - 1.2 = 0.64 м < d_safe=2.0
        cfg.gnron_fix = True
        return (np.array([0.0, 0.0]), np.array([8.8, 8.8]),
                [CircleObstacle(center=np.array([4.0, 5.0]), radius=1.0),
                 CircleObstacle(center=np.array([7.5, 7.5]), radius=1.2)])

    raise ValueError(f"Неизвестный сценарий: {name}")

MODE_PRESETS = {
    "basic":      {"label": "Вариант 1: простое движение"},
    "normalized": {"label": "Вариант 2: нормализация + EMA",
                   "normalize": True, "smoothing": 0.6},
    "gains":      {"label": "Вариант 3: усиление у опасной зоны",
                   "adaptive_rep": True, "adaptive_alpha": 2.0, "k_rep": 3.0},
    "dynamic":    {"label": "Вариант 4: динамические препятствия",
                   "dynamic": True, "prediction_horizon": 5.0},
}

SCENARIOS = {
    "base":  "Базовый (круги)",
    "trap":  "Ловушка",
    "wall":  "Стенка (локальный минимум)",
    "mixed": "Смешанный (круг + прямоуг. + 5-уг.)",
    "gnron": "Цель у препятствия (GNRON)",
}

POTENTIALS = {
    "quadratic": "Квадратичное притяжение",
    "linear":    "Линейное притяжение",
    "hybrid":    "Гибридное притяжение",
}

FORMATIONS = {
    "none":    "Одиночный робот",
    "chain":   "Цепь (3 робота, п.9)",
    "pursuit": "Гонка за лидером (3 робота, п.9)",
    "search":  "Свободный поиск (3 робота, п.9)",
}

N_GROUP = 3     # число роботов в распределённой системе


# ============================================================================
#  tkinter-панель
# ============================================================================
class ControlPanel:
    def __init__(self, cmd_queue: queue.Queue):
        self.cmd_queue = cmd_queue
        self.root = tk.Tk()
        self.root.title("Панель управления APF")
        self.root.geometry("400x830")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()

    def _build(self):
        ttk.Label(self.root, text="Режим (доп. задание)",
                  font=("", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
        for mode, info in MODE_PRESETS.items():
            ttk.Button(self.root, text=info["label"],
                       command=lambda m=mode: self._send("mode", m)
                       ).pack(fill="x", padx=10, pady=2)

        ttk.Separator(self.root).pack(fill="x", padx=10, pady=8)

        ttk.Label(self.root, text="Сценарий",
                  font=("", 11, "bold")).pack(anchor="w", padx=10, pady=(0, 2))
        for sc, label in SCENARIOS.items():
            ttk.Button(self.root, text=label,
                       command=lambda s=sc: self._send("scenario", s)
                       ).pack(fill="x", padx=10, pady=2)

        ttk.Separator(self.root).pack(fill="x", padx=10, pady=8)

        ttk.Label(self.root, text="Потенциал притяжения (п.7)",
                  font=("", 11, "bold")).pack(anchor="w", padx=10, pady=(0, 2))
        for pot, label in POTENTIALS.items():
            ttk.Button(self.root, text=label,
                       command=lambda p=pot: self._send("potential", p)
                       ).pack(fill="x", padx=10, pady=2)

        ttk.Separator(self.root).pack(fill="x", padx=10, pady=8)

        ttk.Label(self.root, text="Формация (п.9)",
                  font=("", 11, "bold")).pack(anchor="w", padx=10, pady=(0, 2))
        for f, label in FORMATIONS.items():
            ttk.Button(self.root, text=label,
                       command=lambda fv=f: self._send("formation", fv)
                       ).pack(fill="x", padx=10, pady=2)

        ttk.Separator(self.root).pack(fill="x", padx=10, pady=8)

        ttk.Label(self.root, text="Визуализация полей",
                  font=("", 11, "bold")).pack(anchor="w", padx=10, pady=(0, 2))
        ttk.Button(self.root, text="Векторное поле (F_total)",
                   command=lambda: self._send("vector_field", "total")
                   ).pack(fill="x", padx=10, pady=2)
        ttk.Button(self.root, text="Поля F_att (зелёный) и F_rep (красный)",
                   command=lambda: self._send("vector_field", "split")
                   ).pack(fill="x", padx=10, pady=2)
        ttk.Button(self.root, text="Скрыть поля",
                   command=lambda: self._send("vector_field", "none")
                   ).pack(fill="x", padx=10, pady=2)

        ttk.Separator(self.root).pack(fill="x", padx=10, pady=8)

        self.status = tk.StringVar(value="симуляция запущена")
        ttk.Label(self.root, textvariable=self.status,
                  foreground="#555", wraplength=360
                  ).pack(anchor="w", padx=10, pady=(0, 10))

    def _send(self, kind: str, value):
        self.cmd_queue.put((kind, value))
        self.status.set(f"команда: {kind} = {value}")

    def _on_close(self):
        self.cmd_queue.put(("quit", None))
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ============================================================================
#  Симуляция (matplotlib)
# ============================================================================
class Simulation:
    def __init__(self, cmd_queue: queue.Queue):
        self.cmd_queue = cmd_queue
        self.cfg = Config()
        self.mode = "basic"
        self.scenario = "base"
        self.potential = "quadratic"
        self.formation = "none"
        self.obstacles: list = []
        self.start = np.zeros(2)
        self.goal = np.zeros(2)
        self.robots: list[RobotState] = []
        self.tick = 0
        self.running = True
        self.vector_field_mode = "none"   # none | total | split
        self.obstacle_tracks: list[list] = []
        # состояние «свободного поиска»
        self.goal_known = False           # цель обнаружена кем-то из группы
        self.found_tick: int | None = None
        self.found_by: int | None = None
        self.slots: list = []
        self.search_rng = np.random.default_rng(0)

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.fig.canvas.manager.set_window_title("APF-симуляция")
        self.fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.08)
        self._reset()

    def _apply_preset(self):
        self.cfg = Config()
        preset = MODE_PRESETS[self.mode]
        for k, v in preset.items():
            if k != "label":
                setattr(self.cfg, k, v)
        self.cfg.potential = self.potential
        if self.formation == "search":
            # цель заранее неизвестна — на обследование нужно больше тактов
            self.cfg.max_steps = self.cfg.search_max_steps

    def _reset(self):
        self._apply_preset()
        self.start, self.goal, self.obstacles = build_scenario(self.scenario, self.cfg)
        self.obstacle_tracks = [[o.centroid().copy() for o in self.obstacles]]

        if self.formation in ("chain", "pursuit"):
            starts = convoy_starts(self.start, self.goal, self.cfg, N_GROUP,
                                   self.cfg.chain_spacing)
            self.robots = [RobotState(name=f"R{i}({'L' if i==0 else 'F'})", pos=p)
                           for i, p in enumerate(starts)]
        elif self.formation == "search":
            starts = convoy_starts(self.start, self.goal, self.cfg, N_GROUP,
                                   self.cfg.chain_spacing)
            self.robots = [RobotState(name=f"R{i}", pos=p) for i, p in enumerate(starts)]
        else:
            self.robots = [RobotState(name="R", pos=self.start.copy())]

        self.tick = 0
        self.goal_known = self.formation != "search"
        self.found_tick = None
        self.found_by = None
        self.slots = []
        self.search_rng = np.random.default_rng(self.cfg.seed)
        if self.formation == "search":
            for i, r in enumerate(self.robots):
                self._assign_waypoint(i, r)

        self.tick = 0
        self.running = True
        self._draw()

    def _draw_vector_field(self):
        """Визуализация векторного поля сил."""
        if self.vector_field_mode == "none":
            return

        n = 22
        xs = np.linspace(0.5, self.cfg.width - 0.5, n)
        ys = np.linspace(0.5, self.cfg.height - 0.5, n)
        X, Y = np.meshgrid(xs, ys)

        if self.vector_field_mode == "total":
            Ux, Uy = np.zeros_like(X), np.zeros_like(Y)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pos = np.array([X[i, j], Y[i, j]])
                    F, _, _ = total_force(pos, self.goal, self.obstacles, self.cfg)
                    n_f = float(np.linalg.norm(F))
                    if n_f > EPS:
                        Ux[i, j], Uy[i, j] = F[0] / n_f, F[1] / n_f
            self.ax.quiver(X, Y, Ux, Uy, color="#1f77b4", alpha=0.6,
                          scale=30, width=0.003, zorder=1)
        elif self.vector_field_mode == "split":
            Ux_a, Uy_a = np.zeros_like(X), np.zeros_like(Y)
            Ux_r, Uy_r = np.zeros_like(X), np.zeros_like(Y)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    pos = np.array([X[i, j], Y[i, j]])
                    F_a = attractive_force(pos, self.goal, self.cfg)
                    F_r = np.zeros(2)
                    for obs in self.obstacles:
                        F_r += repulsive_force(pos, obs, self.cfg, self.goal)
                    n_a = float(np.linalg.norm(F_a))
                    n_r = float(np.linalg.norm(F_r))
                    if n_a > EPS:
                        Ux_a[i, j], Uy_a[i, j] = F_a[0] / n_a, F_a[1] / n_a
                    if n_r > EPS:
                        Ux_r[i, j], Uy_r[i, j] = F_r[0] / n_r, F_r[1] / n_r
            self.ax.quiver(X, Y, Ux_a, Uy_a, color="#2ca02c", alpha=0.6,
                          scale=30, width=0.003, zorder=1, label="F_att")
            self.ax.quiver(X, Y, Ux_r, Uy_r, color="#d62728", alpha=0.6,
                          scale=30, width=0.003, zorder=1, label="F_rep")

    def _draw(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.cfg.width)
        self.ax.set_ylim(0, self.cfg.height)
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.25, ls=":")
        self.ax.set_xlabel("x, м")
        self.ax.set_ylabel("y, м")

        reached = all(r.done for r in self.robots)
        if not self.running:
            if reached:
                status = {"chain": "✓ ЦЕПЬ ПРИБЫЛА",
                          "pursuit": "✓ ГРУППА ПРИБЫЛА",
                          "search": "✓ ГРУППА СОБРАНА У ЦЕЛИ"}.get(
                              self.formation, "✓ ЦЕЛЬ ДОСТИГНУТА")
            else:
                status = "✗ ЛИМИТ ИТЕРАЦИЙ"
        else:
            status = ""
        if self.formation == "search":
            found = (f"цель найдена R{self.found_by} на такте {self.found_tick}"
                     if self.goal_known else "идёт поиск цели")
            status = f"\n{found}  {status}"
        elif self.formation == "pursuit" and status:
            status = f"\n{status}"

        self.ax.set_title(
            f"{MODE_PRESETS[self.mode]['label']}  |  сценарий: {self.scenario}  |  "
            f"потенциал: {self.potential}  |  такт: {self.tick}  {status}"
        )

        # Векторное поле
        self._draw_vector_field()

        # Безопасные зоны (п.5): граница, на которой центр робота
        # начинает испытывать отталкивание (d_clear = d_safe)
        gx = np.linspace(0, self.cfg.width, 100)
        gy = np.linspace(0, self.cfg.height, 100)
        GX, GY = np.meshgrid(gx, gy)
        for o in self.obstacles:
            level = o.safe_zone(self.cfg) + self.cfg.robot_radius
            self.ax.contour(GX, GY, o.distance_field(GX, GY), levels=[level],
                            colors="#7f7f7f", linestyles="--", linewidths=0.9,
                            alpha=0.7, zorder=2)
        if self.obstacles:
            self.ax.plot([], [], "--", color="#7f7f7f", lw=0.9, label="безопасная зона")

        # Препятствия
        for o in self.obstacles:
            o.draw(self.ax, facecolor="#9aa5b1", edgecolor="#333",
                   lw=1.2, zorder=3, alpha=0.9)

        # Траектории препятствий (для dynamic)
        if self.cfg.dynamic and len(self.obstacle_tracks) > 1:
            tracks = np.array(self.obstacle_tracks)
            for j in range(tracks.shape[1]):
                self.ax.plot(tracks[:, j, 0], tracks[:, j, 1],
                            color="#1f77b4", lw=0.8, ls=":", alpha=0.6, zorder=2)

        # Траектории роботов
        colors = ["#d62728", "#ff7f0e", "#9467bd"]
        for i, r in enumerate(self.robots):
            p = np.asarray(r.path)
            self.ax.plot(p[:, 0], p[:, 1], color=colors[i % len(colors)], lw=1.8,
                         label=r.name, zorder=5)
            self.ax.add_patch(CirclePatch(r.pos, self.cfg.robot_radius,
                                          color=colors[i % len(colors)],
                                          alpha=0.6, zorder=6))
            self.ax.plot(*r.pos, "o", color=colors[i % len(colors)], ms=5, zorder=7)

        # Линии связи в формации «цепь»
        if self.formation == "chain" and len(self.robots) > 1:
            pts = np.array([r.pos for r in self.robots])
            self.ax.plot(pts[:, 0], pts[:, 1], "-", color="#555", lw=0.8,
                         alpha=0.6, zorder=6)

        # Линии связи «гонки за лидером»: каждый ведомый связан с лидером
        if self.formation == "pursuit" and len(self.robots) > 1:
            lp = self.robots[0].pos
            for r in self.robots[1:]:
                self.ax.plot([lp[0], r.pos[0]], [lp[1], r.pos[1]], "-",
                             color="#555", lw=0.8, alpha=0.6, zorder=6)

        # «Свободный поиск»: полосы обследования, радиусы датчиков,
        # пройденные и текущие точки обследования, места сбора
        if self.formation == "search":
            n = len(self.robots)
            for k in range(1, n):
                x = search_strip(k, n, self.cfg)[0]
                self.ax.axvline(x, color="#999", lw=0.8, ls="-.", alpha=0.6, zorder=1)
            for i, r in enumerate(self.robots):
                c = colors[i % len(colors)]
                self.ax.add_patch(CirclePatch(r.pos, self.cfg.sensor_range, fill=False,
                                              ec=c, lw=0.8, ls="--", alpha=0.5, zorder=4))
                if r.visited:
                    v = np.array(r.visited)
                    self.ax.plot(v[:, 0], v[:, 1], "x", color=c, ms=5,
                                 alpha=0.5, zorder=4)
                if not self.goal_known and r.waypoint is not None:
                    self.ax.plot(*r.waypoint, "x", color=c, ms=9, mew=2, zorder=7)
            self.ax.plot([], [], "--", color="#999", lw=0.8, label="радиус датчика")
            self.ax.plot([], [], "x", color="#555", ms=7, label="точки обследования")
            if self.goal_known:
                for p in self.slots:
                    self.ax.plot(*p, "s", ms=6, mfc="none", mec="#333", zorder=7)
                self.ax.plot([], [], "s", ms=6, mfc="none", mec="#333",
                             label="места сбора")

        # Старт / цель
        self.ax.plot(*self.start, "o", color="green", ms=9, zorder=8, label="старт")
        if self.formation == "search" and not self.goal_known:
            # цель роботам ещё неизвестна — показываем контуром
            self.ax.plot(*self.goal, "*", color="none", ms=16, mec="k",
                         zorder=8, label="цель (не найдена)")
        else:
            self.ax.plot(*self.goal, "*", color="gold", ms=16, mec="k",
                         zorder=8, label="цель")

        self.ax.legend(loc="upper left", fontsize=8)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    #  Вспомогательные методы распределённых систем
    # ------------------------------------------------------------------
    def _settled(self, r: RobotState, tgt, tol: float) -> bool:
        """Робот на своём месте: подъехал на tol или остановился рядом с ним."""
        d_tgt = distance(r.pos, tgt)
        step = distance(r.path[-1], r.path[-2]) if len(r.path) > 1 else 0.0
        return d_tgt <= tol or (d_tgt <= self.cfg.formation_tol and
                                step < self.cfg.settle_tol)

    def _assign_waypoint(self, i: int, r: RobotState) -> None:
        """Назначить роботу i новую точку обследования в его полосе."""
        if r.waypoint is not None:
            r.visited.append(r.waypoint.copy())
        r.waypoint = new_waypoint(i, len(self.robots), r, self.obstacles,
                                  self.cfg, self.search_rng)
        r.waypoint_tick = self.tick
        r.waypoint_escapes = r.escapes

    def _search_update(self) -> None:
        """Такт «свободного поиска»: обнаружение цели и смена точек обследования."""
        if not self.goal_known:
            # Обнаружение: цель попала в радиус датчика одного из роботов.
            # Робот сразу сообщает координаты цели всей группе.
            for i, r in enumerate(self.robots):
                if distance(r.pos, self.goal) <= self.cfg.sensor_range:
                    self.goal_known = True
                    self.found_tick = self.tick
                    self.found_by = i
                    others = [j for j in range(len(self.robots)) if j != i]
                    # ближайшие к цели роботы получают ближайшие места сбора
                    others.sort(key=lambda j: distance(self.robots[j].pos, self.goal))
                    slots = gather_slots(self.goal, len(self.robots),
                                         self.obstacles, self.cfg)
                    self.slots = [None] * len(self.robots)
                    self.slots[i] = slots[0]
                    for k, j in enumerate(others):
                        self.slots[j] = slots[k + 1]
                    for rr in self.robots:      # сброс состояния манёвра выхода
                        rr.history.clear()
                        rr.escape_left = 0
                    return
            # Смена точки обследования: достигнута, робот застрял или истекло время
            for i, r in enumerate(self.robots):
                if (distance(r.pos, r.waypoint) <= self.cfg.waypoint_tol or
                        r.escapes - r.waypoint_escapes >= 2 or
                        self.tick - r.waypoint_tick >= self.cfg.waypoint_timeout):
                    self._assign_waypoint(i, r)

    def _step(self):
        if not self.running:
            return

        # Движение препятствий (вариант 4)
        if self.cfg.dynamic:
            for o in self.obstacles:
                o.advance(1.0, self.cfg)
            self.obstacle_tracks.append([o.centroid().copy() for o in self.obstacles])

        # Упреждение положения препятствий
        obs_calc = ([o.predicted(self.cfg.prediction_horizon) for o in self.obstacles]
                    if self.cfg.dynamic else self.obstacles)

        rng = np.random.default_rng(self.cfg.seed + self.tick)

        chain = self.formation == "chain" and len(self.robots) > 1
        pursuit = self.formation == "pursuit" and len(self.robots) > 1
        search = self.formation == "search"

        if search:
            self._search_update()

        for i, r in enumerate(self.robots):
            # Прибывший робот продолжает удерживать свою точку по APF:
            # так он уклоняется от движущихся препятствий, а не стоит на месте
            # Целевая точка зависит от подхода:
            #   цепь    — лидер едет в цель, ведомый — по пути предыдущего робота;
            #   гонка   — лидер едет в цель, ведомые — прямо к текущему положению лидера;
            #   поиск   — пока цель не найдена, каждый обследует свою полосу,
            #             после обнаружения все собираются у цели.
            if search:
                tgt = self.slots[i] if self.goal_known else r.waypoint
            elif i == 0 or not (chain or pursuit):
                tgt = self.goal
            elif chain:
                tgt = breadcrumb(self.robots[i - 1].path, self.cfg.chain_spacing)
                if tgt is None:             # предыдущий ещё не отъехал — ждём
                    continue
            else:
                tgt = pursuit_point(self.robots[0].pos, r.pos, self.cfg.pursuit_spacing)
            # Межроботное отталкивание для группы роботов
            neighbours = []
            if self.formation != "none":
                neighbours = [CircleObstacle(center=o.pos, radius=self.cfg.robot_radius,
                                             d_safe_override=self.cfg.inter_robot_safe)
                              for o in self.robots if o is not r]
            delta, _ = compute_step(r, tgt, list(obs_calc) + neighbours, self.cfg, rng)
            _advance_state(r, delta, tgt, list(self.obstacles) + neighbours, self.cfg)

            # Проверка прибытия (флаг запоминается)
            if r.done:
                continue
            if search:
                if self.goal_known:
                    r.done = self._settled(r, tgt, self.cfg.goal_tol)
            elif i == 0:
                r.done = distance(r.pos, self.goal) <= self.cfg.goal_tol
            elif self.robots[i - 1 if chain else 0].done:
                # ведомый прибыл, когда остановился предыдущий (в цепи)
                # или лидер (в гонке), а сам он подъехал к своему месту
                # в строю и перестал двигаться
                r.done = self._settled(r, tgt, self.cfg.follower_tol)

        self.tick += 1

        # Критерий остановки: прибыли все роботы
        if all(r.done for r in self.robots):
            self.running = False
        elif self.tick >= self.cfg.max_steps:
            self.running = False

        self._draw()

    def _poll_commands(self):
        changed = False
        try:
            while True:
                kind, value = self.cmd_queue.get_nowait()
                if kind == "quit":
                    plt.close(self.fig)
                    return False
                elif kind == "mode":
                    if value != self.mode:
                        self.mode = value
                        changed = True
                elif kind == "scenario":
                    if value != self.scenario:
                        self.scenario = value
                        changed = True
                elif kind == "potential":
                    if value != self.potential:
                        self.potential = value
                        changed = True
                elif kind == "formation":
                    if value != self.formation:
                        self.formation = value
                        changed = True
                elif kind == "vector_field":
                    self.vector_field_mode = value
                    self._draw()
        except queue.Empty:
            pass

        if changed:
            self._reset()
        return True

    def run(self):
        while self._poll_commands():
            self._step()
            plt.pause(0.05)
        plt.close(self.fig)


# ============================================================================
#  Точка входа
# ============================================================================
def main():
    cmd_queue = queue.Queue()

    panel = ControlPanel(cmd_queue)
    t = threading.Thread(target=panel.run, daemon=True)
    t.start()

    sim = Simulation(cmd_queue)
    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            panel.root.after(0, panel.root.destroy)
        except Exception:
            pass


if __name__ == "__main__":
    main()