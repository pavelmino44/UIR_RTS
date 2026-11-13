import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.animation import FuncAnimation

# ---------------------------------------------------------------------------
# 4. Параметры рабочей области (исходные значения сохранены)
# ---------------------------------------------------------------------------
area_bounds = (0, 0, 100, 100)
start = (10, 10)
goal = (90, 90)

# Типы препятствий:
#   прямоугольник : (x_min, y_min, x_max, y_max)            - исходный формат
#   круг          : ("circle", cx, cy, r)
#   многоугольник : ("polygon", [(x1, y1), (x2, y2), ...])
obstacles = [
    (20, 20, 40, 60),
    (60, 40, 80, 80),
    (30, 75, 70, 90)
]
max_iter = 1000
step_size = 5
goal_radius = 5

# Дополнительные карты для тестирования (смешанные типы препятствий)
obstacles_mixed = [
    (15, 40, 45, 50),
    ("circle", 70, 30, 12),
    ("circle", 50, 70, 10),
    ("polygon", [(60, 55), (85, 60), (80, 85), (62, 80)]),
    ("polygon", [(20, 65), (35, 60), (40, 80), (22, 88)]),
]
obstacles_maze = [
    (0, 30, 70, 38),
    (30, 62, 100, 70),
    ("circle", 85, 20, 8),
    ("polygon", [(10, 80), (25, 78), (20, 95)]),
]
MAPS = {
    "Карта 1 (прямоугольники)": obstacles,
    "Карта 2 (смешанные)": obstacles_mixed,
    "Карта 3 (коридоры)": obstacles_maze,
}

# Статистика последнего запуска базового RRT (число итераций)
stats = {"iterations": 0}


# ---------------------------------------------------------------------------
# 5. Класс вершины дерева
# ---------------------------------------------------------------------------
class Vertex:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = 0.0


# ---------------------------------------------------------------------------
# Геометрия: проверка столкновений отрезка с препятствиями разных типов
# ---------------------------------------------------------------------------
def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _orient(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment(p, q, r):
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def segments_intersect(p1, p2, p3, p4):
    """Пересечение отрезков p1p2 и p3p4 (включая касание)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def point_in_polygon(pt, poly):
    """Алгоритм трассировки луча."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


def segment_hits_rect(p1, p2, rect):
    """Отрезок и прямоугольник (метод Лианга-Барски)."""
    x_min, y_min, x_max, y_max = rect
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p1[0] - x_min), (dx, x_max - p1[0]),
                 (-dy, p1[1] - y_min), (dy, y_max - p1[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


def segment_hits_circle(p1, p2, cx, cy, r):
    """Расстояние от центра круга до отрезка не больше радиуса."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        t = 0.0
    else:
        t = ((cx - p1[0]) * dx + (cy - p1[1]) * dy) / length2
        t = max(0.0, min(1.0, t))
    px, py = p1[0] + t * dx, p1[1] + t * dy
    return math.hypot(px - cx, py - cy) <= r


def segment_hits_polygon(p1, p2, poly):
    if point_in_polygon(p1, poly) or point_in_polygon(p2, poly):
        return True
    n = len(poly)
    for i in range(n):
        if segments_intersect(p1, p2, poly[i], poly[(i + 1) % n]):
            return True
    return False


def segment_hits_obstacle(p1, p2, obs):
    if obs[0] == "circle":
        return segment_hits_circle(p1, p2, obs[1], obs[2], obs[3])
    if obs[0] == "polygon":
        return segment_hits_polygon(p1, p2, obs[1])
    return segment_hits_rect(p1, p2, obs)


# ---------------------------------------------------------------------------
# Базовые функции RRT
# ---------------------------------------------------------------------------
def generate_random_point(area_bounds, goal=None, goal_bias=0.0):
    """Случайная точка в рабочей области (с небольшой вероятностью - цель)."""
    if goal is not None and random.random() < goal_bias:
        return goal
    x_min, y_min, x_max, y_max = area_bounds
    return (random.uniform(x_min, x_max), random.uniform(y_min, y_max))


def find_nearest_vertex(vertices, point):
    best = vertices[0]
    best_d = (best.x - point[0]) ** 2 + (best.y - point[1]) ** 2
    for v in vertices[1:]:
        d = (v.x - point[0]) ** 2 + (v.y - point[1]) ** 2
        if d < best_d:
            best, best_d = v, d
    return best


def steer(nearest, random_point, step_size):
    """Новая вершина на расстоянии не более step_size от nearest."""
    dx = random_point[0] - nearest.x
    dy = random_point[1] - nearest.y
    d = math.hypot(dx, dy)
    if d <= step_size:
        new_x, new_y = random_point[0], random_point[1]
    else:
        new_x = nearest.x + step_size * dx / d
        new_y = nearest.y + step_size * dy / d
    new_vertex = Vertex(new_x, new_y, nearest)
    new_vertex.cost = nearest.cost + math.hypot(new_x - nearest.x,
                                                new_y - nearest.y)
    return new_vertex


def is_collision_free(v1, v2, obstacles):
    """True, если отрезок v1-v2 не пересекает ни одно препятствие."""
    p1, p2 = (v1.x, v1.y), (v2.x, v2.y)
    for obs in obstacles:
        if segment_hits_obstacle(p1, p2, obs):
            return False
    return True


# ---------------------------------------------------------------------------
# 6. Планирование RRT
# ---------------------------------------------------------------------------
def rrt_plan(start, goal, obstacles, area_bounds, max_iter, step_size, goal_radius):
    start_vertex = Vertex(start[0], start[1])
    vertices = [start_vertex]
    goal_v = Vertex(goal[0], goal[1])
    stats["iterations"] = max_iter

    for i in range(max_iter):
        random_point = generate_random_point(area_bounds, goal, 0.05)
        nearest = find_nearest_vertex(vertices, random_point)
        new_vertex = steer(nearest, random_point, step_size)
        if is_collision_free(nearest, new_vertex, obstacles):
            vertices.append(new_vertex)
            # проверка достижения цели
            if distance(new_vertex, goal_v) <= goal_radius:
                if is_collision_free(new_vertex, goal_v, obstacles):
                    goal_v.parent = new_vertex
                    goal_v.cost = new_vertex.cost + distance(new_vertex, goal_v)
                    vertices.append(goal_v)
                    stats["iterations"] = i + 1
                    return goal_v, vertices
    return None, vertices


# ---------------------------------------------------------------------------
# 7. Восстановление пути
# ---------------------------------------------------------------------------
def extract_path(goal_vertex):
    path = []
    current = goal_vertex
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]


def path_length(path):
    if not path:
        return float("inf")
    return sum(math.hypot(path[i + 1][0] - path[i][0],
                          path[i + 1][1] - path[i][1])
               for i in range(len(path) - 1))


# ---------------------------------------------------------------------------
# 9. RRT*
# ---------------------------------------------------------------------------
class RRTStar:
    def __init__(self, start, goal, obstacles, area_bounds,
                 max_iter=1000, step_size=10, neighbor_radius=20,
                 goal_radius=5, record_every=10):
        self.start = Vertex(start[0], start[1])
        self.goal = Vertex(goal[0], goal[1])
        self.obstacles = obstacles
        self.area_bounds = area_bounds
        self.max_iter = max_iter
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.goal_radius = goal_radius
        self.vertices = [self.start]
        # для анимации и метрик
        self.record_every = record_every
        self.history = []          # снимки рёбер дерева
        self.first_goal_iter = None  # итерация первого достижения цели
        self.iterations = 0

    def _edges(self):
        return [((v.parent.x, v.parent.y), (v.x, v.y))
                for v in self.vertices if v.parent is not None]

    def plan(self):
        """Основной цикл RRT*. Возвращает вершину цели или None."""
        for i in range(1, self.max_iter + 1):
            self.iterations = i
            rp = generate_random_point(self.area_bounds,
                                       (self.goal.x, self.goal.y), 0.05)
            nearest = find_nearest_vertex(self.vertices, rp)
            new_vertex = steer(nearest, rp, self.step_size)
            if is_collision_free(nearest, new_vertex, self.obstacles):
                near_vertices = self.find_near_vertices(new_vertex)
                self.choose_parent(new_vertex, near_vertices)
                self.vertices.append(new_vertex)
                self.rewire(new_vertex, near_vertices)
                if (self.first_goal_iter is None and
                        distance(new_vertex, self.goal) <= self.goal_radius and
                        is_collision_free(new_vertex, self.goal, self.obstacles)):
                    self.first_goal_iter = i
            if i % self.record_every == 0:
                self.history.append(self._edges())

        # выбор лучшей вершины рядом с целью (после всех переподключений)
        best, best_cost = None, float("inf")
        for v in self.vertices:
            d = distance(v, self.goal)
            if d <= self.goal_radius and v.cost + d < best_cost:
                if is_collision_free(v, self.goal, self.obstacles):
                    best, best_cost = v, v.cost + d
        if best is None:
            self.history.append(self._edges())
            return None
        self.goal.parent = best
        self.goal.cost = best_cost
        self.vertices.append(self.goal)
        self.history.append(self._edges())
        return self.goal

    def find_near_vertices(self, vertex):
        """Вершины дерева в радиусе neighbor_radius от новой вершины."""
        r2 = self.neighbor_radius ** 2
        return [v for v in self.vertices
                if (v.x - vertex.x) ** 2 + (v.y - vertex.y) ** 2 <= r2]

    def choose_parent(self, new_vertex, near_vertices):
        """Выбрать родителя, дающего минимальную стоимость пути."""
        best_parent = new_vertex.parent
        best_cost = new_vertex.cost
        for v in near_vertices:
            c = v.cost + distance(v, new_vertex)
            if c < best_cost and is_collision_free(v, new_vertex, self.obstacles):
                best_parent, best_cost = v, c
        new_vertex.parent = best_parent
        new_vertex.cost = best_cost
        return new_vertex

    def _propagate_cost(self, vertex):
        """Обновить стоимость потомков после смены родителя."""
        for u in self.vertices:
            if u.parent is vertex:
                u.cost = vertex.cost + distance(vertex, u)
                self._propagate_cost(u)

    def rewire(self, new_vertex, near_vertices):
        """Переподключить соседей, если путь через new_vertex короче."""
        for v in near_vertices:
            if v is new_vertex.parent or v is self.start:
                continue
            c = new_vertex.cost + distance(new_vertex, v)
            if c < v.cost and is_collision_free(new_vertex, v, self.obstacles):
                v.parent = new_vertex
                v.cost = c
                self._propagate_cost(v)

    def extract_path(self, goal_vertex=None):
        if goal_vertex is None:
            goal_vertex = self.goal if self.goal.parent is not None else None
        if goal_vertex is None:
            return None
        return extract_path(goal_vertex)


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def draw_obstacles(ax, obstacles):
    for obs in obstacles:
        if obs[0] == "circle":
            ax.add_patch(Circle((obs[1], obs[2]), obs[3], color="gray", alpha=0.8))
        elif obs[0] == "polygon":
            ax.add_patch(Polygon(obs[1], closed=True, color="gray", alpha=0.8))
        else:
            x0, y0, x1, y1 = obs
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   color="gray", alpha=0.8))


def setup_axes(ax, obstacles, start, goal, title=""):
    ax.set_xlim(area_bounds[0], area_bounds[2])
    ax.set_ylim(area_bounds[1], area_bounds[3])
    ax.set_aspect("equal")
    draw_obstacles(ax, obstacles)
    ax.plot(start[0], start[1], "go", markersize=10, label="start")
    ax.plot(goal[0], goal[1], "r*", markersize=14, label="goal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def visualize(vertices, obstacles, start, goal, path, title="RRT", ax=None,
              show=True):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7, 7))
    setup_axes(ax, obstacles, start, goal, title)
    for v in vertices:
        if v.parent is not None:
            ax.plot([v.parent.x, v.x], [v.parent.y, v.y], "-",
                    color="steelblue", linewidth=0.6)
    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, "r-", linewidth=2.5, label="путь")
    ax.legend(loc="upper left")
    if own and show:
        plt.show()


def history_from_vertices(vertices, frames=60):
    """Снимки роста дерева RRT (рёбра добавляются по порядку)."""
    n = len(vertices)
    step = max(1, n // frames)
    history = []
    for k in range(step, n + step, step):
        sub = vertices[:min(k, n)]
        history.append([((v.parent.x, v.parent.y), (v.x, v.y))
                        for v in sub if v.parent is not None])
    return history


def animate_tree(history, obstacles, start, goal, path, title="",
                 interval=60, save_path=None):
    """Анимация построения дерева. history - список снимков рёбер."""
    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axes(ax, obstacles, start, goal, title)
    lc = LineCollection([], colors="steelblue", linewidths=0.6)
    ax.add_collection(lc)
    path_line, = ax.plot([], [], "r-", linewidth=2.5)
    text = ax.text(2, 96, "", fontsize=10)
    n_frames = len(history) + 15  # последние кадры - показ итогового пути

    def update(frame):
        k = min(frame, len(history) - 1)
        lc.set_segments(history[k])
        text.set_text(f"кадр {k + 1}/{len(history)}, рёбер: {len(history[k])}")
        if frame >= len(history) and path:
            xs, ys = zip(*path)
            path_line.set_data(xs, ys)
        return lc, path_line, text

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval,
                         blit=False, repeat=False)
    if save_path:
        try:
            anim.save(save_path, writer="pillow", fps=15)
            print(f"Анимация сохранена: {save_path}")
        except Exception as e:
            print("Не удалось сохранить анимацию:", e)
    plt.show()
    return anim


# ---------------------------------------------------------------------------
# Метрики и сравнение RRT и RRT*
# ---------------------------------------------------------------------------
def run_rrt(obstacles, seed=None, **kw):
    p = dict(max_iter=max_iter, step_size=step_size, goal_radius=goal_radius)
    p.update(kw)
    if seed is not None:
        random.seed(seed)
    t0 = time.perf_counter()
    gv, vertices = rrt_plan(start, goal, obstacles, area_bounds,
                            p["max_iter"], p["step_size"], p["goal_radius"])
    elapsed = time.perf_counter() - t0
    path = extract_path(gv) if gv else None
    return {"found": gv is not None, "path": path, "length": path_length(path),
            "time": elapsed, "iterations": stats["iterations"],
            "vertices": len(vertices), "tree": vertices}


def run_rrt_star(obstacles, seed=None, **kw):
    p = dict(max_iter=max_iter, step_size=step_size, neighbor_radius=20,
             goal_radius=goal_radius)
    p.update(kw)
    if seed is not None:
        random.seed(seed)
    t0 = time.perf_counter()
    planner = RRTStar(start, goal, obstacles, area_bounds,
                      max_iter=p["max_iter"], step_size=p["step_size"],
                      neighbor_radius=p["neighbor_radius"],
                      goal_radius=p["goal_radius"])
    gv = planner.plan()
    elapsed = time.perf_counter() - t0
    path = planner.extract_path(gv)
    return {"found": gv is not None, "path": path, "length": path_length(path),
            "time": elapsed, "iterations": planner.iterations,
            "vertices": len(planner.vertices), "tree": planner.vertices,
            "planner": planner}


def print_comparison(name, r_rrt, r_star):
    print(f"\n{name}")
    print(f"{'Метрика':<22}{'RRT':>12}{'RRT*':>12}")
    print("-" * 46)
    print(f"{'Путь найден':<22}{str(r_rrt['found']):>12}{str(r_star['found']):>12}")
    print(f"{'Длина пути':<22}{r_rrt['length']:>12.2f}{r_star['length']:>12.2f}")
    print(f"{'Время, с':<22}{r_rrt['time']:>12.3f}{r_star['time']:>12.3f}")
    print(f"{'Число итераций':<22}{r_rrt['iterations']:>12d}{r_star['iterations']:>12d}")
    print(f"{'Число вершин':<22}{r_rrt['vertices']:>12d}{r_star['vertices']:>12d}")


def compare_on_maps(seed=1, rrt_star_iter=2000):
    """Сравнение RRT и RRT* на всех картах + рисунки."""
    # 2 строки (RRT / RRT*) x 3 столбца (карты): помещается в окно без наложений
    fig, axes = plt.subplots(2, len(MAPS), figsize=(15, 9.5),
                             layout="constrained")
    for col, (name, obs) in enumerate(MAPS.items()):
        r1 = run_rrt(obs, seed=seed, max_iter=3000)
        r2 = run_rrt_star(obs, seed=seed, max_iter=rrt_star_iter,
                          step_size=5, neighbor_radius=15)
        print_comparison(name, r1, r2)
        visualize(r1["tree"], obs, start, goal, r1["path"],
                  title=f"RRT: {name}\nдлина = {r1['length']:.1f}",
                  ax=axes[0][col])
        visualize(r2["tree"], obs, start, goal, r2["path"],
                  title=f"RRT*: {name}\nдлина = {r2['length']:.1f}",
                  ax=axes[1][col])
    plt.savefig("comparison.png", dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# Исследование влияния параметров
# ---------------------------------------------------------------------------
def experiment(param, values, obs=None, seeds=(1, 2, 3, 4, 5)):
    """Усреднение метрик RRT* по нескольким запускам для разных значений param."""
    obs = obs if obs is not None else obstacles
    results = []
    for val in values:
        lengths, times, found = [], [], 0
        for s in seeds:
            kw = {param: val}
            if param == "step_size":
                kw.setdefault("neighbor_radius", max(20, 3 * val))
            r = run_rrt_star(obs, seed=s, **kw)
            times.append(r["time"])
            if r["found"]:
                found += 1
                lengths.append(r["length"])
        results.append({
            "value": val,
            "length": float(np.mean(lengths)) if lengths else float("nan"),
            "time": float(np.mean(times)),
            "success": found / len(seeds),
        })
    return results


def run_parameter_study():
    studies = [
        ("max_iter", [500, 1000, 2000, 3000]),
        ("step_size", [2, 5, 10, 15]),
        ("neighbor_radius", [8, 15, 25, 40]),
    ]
    fig, axes = plt.subplots(len(studies), 2, figsize=(11, 10))
    for row, (param, values) in enumerate(studies):
        res = experiment(param, values)
        print(f"\nВлияние параметра {param} (RRT*, усреднение по 5 запускам)")
        print(f"{param:>16}{'длина':>10}{'время, с':>11}{'успех':>8}")
        for r in res:
            print(f"{r['value']:>16}{r['length']:>10.2f}"
                  f"{r['time']:>11.3f}{r['success']:>8.0%}")
        xs = [r["value"] for r in res]
        axes[row][0].plot(xs, [r["length"] for r in res], "o-")
        axes[row][0].set_xlabel(param)
        axes[row][0].set_ylabel("длина пути")
        axes[row][0].grid(True)
        axes[row][1].plot(xs, [r["time"] for r in res], "s-", color="orange")
        axes[row][1].set_xlabel(param)
        axes[row][1].set_ylabel("время, с")
        axes[row][1].grid(True)
    plt.tight_layout()
    plt.savefig("parameter_study.png", dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# 8. Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(1)

    # --- базовый RRT (исходная часть задания) ---
    goal_vertex, vertices = rrt_plan(
        start, goal, obstacles, area_bounds,
        max_iter, step_size, goal_radius
    )
    path = extract_path(goal_vertex) if goal_vertex else None
    if path:
        print(f"RRT: путь найден, длина = {path_length(path):.2f}, "
              f"вершин = {len(vertices)}, итераций = {stats['iterations']}")
    else:
        print("RRT: путь не найден за", max_iter, "итераций")
    visualize(vertices, obstacles, start, goal, path, title="RRT")

    # --- RRT* ---
    random.seed(1)
    planner = RRTStar(start, goal, obstacles, area_bounds,
                      max_iter=2000, step_size=5, neighbor_radius=15,
                      goal_radius=goal_radius)
    star_goal = planner.plan()
    star_path = planner.extract_path(star_goal)
    if star_path:
        print(f"RRT*: путь найден, длина = {path_length(star_path):.2f}, "
              f"вершин = {len(planner.vertices)}")
    visualize(planner.vertices, obstacles, start, goal, star_path, title="RRT*")

    # --- анимация построения дерева (RRT и RRT*) ---
    animate_tree(history_from_vertices(vertices), obstacles, start, goal, path,
                 title="RRT: рост дерева")
    animate_tree(planner.history, obstacles, start, goal, star_path,
                 title="RRT*: рост дерева", save_path=None)  # "rrt_star.gif"

    # --- сравнение RRT и RRT* на трёх картах (метрики + рисунки) ---
    compare_on_maps()

    # --- исследование параметров max_iter, step_size, neighbor_radius ---
    run_parameter_study()