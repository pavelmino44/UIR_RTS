import heapq
import random
import sys
import time

import matplotlib.pyplot as plt
import networkx as nx
import pygame


# ============================================================
# ВСПОМОГАТЕЛЬНЫЙ КЛАСС ДЛЯ A* (ORIGINAL)
# ============================================================
# Не был явно определён во фрагменте исходного кода, но
# использовался в оригинальной реализации a_star ниже.

class Node:
	def __init__(self, parent=None, position=None):
		self.parent = parent
		self.position = position

		self.g = 0
		self.h = 0
		self.f = 0

	def __eq__(self, other):
		return self.position == other.position


# ============================================================
# ДЕЙКСТРА ДЛЯ ВЗВЕШЕННОГО ГРАФА (ORIGINAL)
# ============================================================

def dijkstra(graph, start):
	distances = {vertex: float('infinity') for vertex in graph}
	distances[start] = 0

	queue = [(0, start)]			# O(log(n))

	while queue:
		current_distance, current_vertex = heapq.heappop(queue)

		if current_distance > distances[current_vertex]:
			continue

		for neighbor, weight in graph[current_vertex].items():
			distance = current_distance + weight

			if distance < distances[neighbor]:
				distances[neighbor] = distance
				heapq.heappush(queue, (distance, neighbor))

	return distances


# ============================================================
# ПОСТРОЕНИЕ ГРАФА (ORIGINAL)
# ============================================================

def build_graph(edges, p):
	graph = {i: {} for i in range(p)}

	return graph


# ============================================================
# ГЕНЕРАЦИЯ ЛАБИРИНТА (ORIGINAL)
# ============================================================

ROWS, COLS = 50, 50


def generate_maze(rows, cols, wall_prob=0.3):
	maze = [
		[
			1 if random.random() < wall_prob else 0
			for _ in range(cols)
		]
		for _ in range(rows)
	]

	maze[0][0] = 0
	maze[rows - 1][cols - 1] = 0

	return maze


# ============================================================
# ЗАПОЛНЕНИЕ ГРАФА РЁБРАМИ (добавлено)
# ============================================================
# build_graph (оригинал) игнорирует edges и возвращает граф без
# единого ребра — поэтому draw_graph рисовал только изолированные
# точки. build_graph не меняется: рёбра добавляются отдельной
# функцией уже после того, как build_graph отработал.

def populate_graph(graph, edges):
	for u, v, weight in edges:
		graph[u][v] = weight
		graph[v][u] = weight

	return graph


# ============================================================
# A* ДЛЯ ВЗВЕШЕННОГО ГРАФА ГОРОДОВ (добавлено)
# ============================================================


def a_star_graph(graph, start, goal, heuristic=None):
	if heuristic is None:
		def heuristic(_node, _goal):
			return 0

	g_score = {start: 0}
	open_heap = [(heuristic(start, goal), start)]
	visited = set()

	while open_heap:
		_, current = heapq.heappop(open_heap)

		if current in visited:
			continue

		visited.add(current)

		if current == goal:
			return g_score[current]

		for neighbor, weight in graph[current].items():
			tentative_g = g_score[current] + weight

			if tentative_g < g_score.get(neighbor, float('inf')):
				g_score[neighbor] = tentative_g

				f_score = tentative_g + heuristic(neighbor, goal)
				heapq.heappush(open_heap, (f_score, neighbor))

	return float('infinity')


# ============================================================
# РАССТОЯНИЯ ДО ВСЕХ ГОРОДОВ (добавлено)
# ============================================================
# Для каждого города считаются кратчайшие расстояния до всех
# остальных городов — один раз алгоритмом Дейкстры (dijkstra
# уже умеет это за один запуск), и один раз алгоритмом A*
# (a_star_graph ищет путь до одной цели, поэтому она вызывается
# отдельно для каждой пары городов).

def all_distances_dijkstra(graph):
	return {
		vertex: dijkstra(graph, vertex)
		for vertex in graph
	}


def all_distances_a_star(graph, heuristic=None):
	distances = {}

	for start in graph:
		distances[start] = {}

		for goal in graph:
			if start == goal:
				distances[start][goal] = 0
			else:
				distances[start][goal] = a_star_graph(
					graph,
					start,
					goal,
					heuristic
				)

	return distances


# ============================================================
# АНАЛИЗ ДОСТИЖИМОСТИ В ПРЕДЕЛАХ MILEAGE (добавлено)
# ============================================================

def count_reachable_within_mileage(distances, mileage):
	# distances: {город: {другой_город: расстояние}}
	# Сам город в подсчёт достижимых не включается.
	return {
		vertex: sum(
			1
			for other, distance in distance_map.items()
			if other != vertex and distance <= mileage
		)
		for vertex, distance_map in distances.items()
	}


def find_city_with_min_reachable(reachable_counts):
	return min(reachable_counts, key=reachable_counts.get)


# ============================================================
# ГАРАНТИЯ ПРОХОДИМОСТИ ЛАБИРИНТА (добавлено)
# ============================================================
# generate_maze не изменяется: она по-прежнему может создать
# лабиринт, где старт и/или финиш замурованы стенами. Вместо
# правки самой функции лабиринт просто перегенерируется, пока
# путь между стартом и финишем действительно не найдётся

def ensure_solvable_maze(rows, cols, wall_prob=0.3, max_attempts=1000):
	start = (0, 0)
	goal = (rows - 1, cols - 1)

	for _ in range(max_attempts):
		maze = generate_maze(rows, cols, wall_prob)

		path, _ = dijkstra_maze(maze, start, goal)

		if path:
			return maze

	raise RuntimeError(
		'Не удалось сгенерировать проходимый лабиринт '
		f'за {max_attempts} попыток. Попробуйте уменьшить wall_prob.'
	)


# ============================================================
# СОСЕДИ КЛЕТКИ (добавлено)
# ============================================================

def get_neighbors(maze, row, col):
	rows = len(maze)
	cols = len(maze[0])

	directions = [
		(-1, 0),
		(1, 0),
		(0, -1),
		(0, 1)
	]

	for dr, dc in directions:
		new_row = row + dr
		new_col = col + dc

		if 0 <= new_row < rows and 0 <= new_col < cols:
			if maze[new_row][new_col] == 0:
				yield new_row, new_col


# ============================================================
# ДЕЙКСТРА ДЛЯ ЛАБИРИНТА (добавлено)
# ============================================================

def dijkstra_maze(maze, start, goal):
	distances = {
		start: 0
	}

	previous = {}

	queue = [(0, start)]
	visited = set()

	while queue:
		current_distance, current = heapq.heappop(queue)

		if current in visited:
			continue

		visited.add(current)

		if current == goal:
			break

		for neighbor in get_neighbors(maze, current[0], current[1]):
			new_distance = current_distance + 1

			if new_distance < distances.get(neighbor, float('inf')):
				distances[neighbor] = new_distance
				previous[neighbor] = current

				heapq.heappush(
					queue,
					(new_distance, neighbor)
				)

	if goal not in distances:
		return [], visited

	path = []
	current = goal

	while current != start:
		path.append(current)
		current = previous[current]

	path.append(start)
	path.reverse()

	return path, visited


# ============================================================
# УСТРАНЕНИЕ ЦИКЛОВ ДЛЯ A* (добавлено)
# ============================================================

def bfs_shortest_path_tree(maze, start):
	previous = {}
	visited = {start}
	queue = [start]
	queue_index = 0

	while queue_index < len(queue):
		current = queue[queue_index]
		queue_index += 1

		for neighbor in get_neighbors(maze, current[0], current[1]):
			if neighbor not in visited:
				visited.add(neighbor)
				previous[neighbor] = current
				queue.append(neighbor)

	return previous


def build_tree_maze(maze, start):
	previous = bfs_shortest_path_tree(maze, start)

	rows = len(maze)
	cols = len(maze[0])

	exp_rows = 2 * rows - 1
	exp_cols = 2 * cols - 1

	tree_maze = [[1] * exp_cols for _ in range(exp_rows)]

	def to_expanded(pos):
		return 2 * pos[0], 2 * pos[1]

	start_row, start_col = to_expanded(start)
	tree_maze[start_row][start_col] = 0

	for node, parent in previous.items():
		node_row, node_col = to_expanded(node)
		tree_maze[node_row][node_col] = 0

		parent_row, parent_col = to_expanded(parent)
		bridge_row = (node_row + parent_row) // 2
		bridge_col = (node_col + parent_col) // 2
		tree_maze[bridge_row][bridge_col] = 0

	return tree_maze


def to_expanded_position(pos):
	return 2 * pos[0], 2 * pos[1]


def collapse_expanded_path(expanded_path):
	# Оставляем только клетки, соответствующие настоящим клеткам
	# исходного лабиринта (координаты-перемычки отбрасываются),
	# чтобы путь можно было отрисовать в исходном масштабе.
	return [
		(row // 2, col // 2)
		for row, col in expanded_path
		if row % 2 == 0 and col % 2 == 0
	]


# ============================================================
# ЗАХВАТ ПОСЕЩЁННЫХ КЛЕТОК A* (добавлено)
# ============================================================
# a_star (оригинал) не возвращает closed_list — только путь.
# Не меняя саму функцию, значение её локальной переменной
# closed_list в момент завершения читается через sys.settrace:
# это просто внешнее наблюдение за уже существующим кодом,
# а не его изменение.

def run_a_star_with_visited(maze, start, end):
	captured_closed = []

	def tracer(frame, event, arg):
		if frame.f_code.co_name == 'a_star':
			if event == 'return':
				closed_list = frame.f_locals.get('closed_list')

				if closed_list is not None:
					captured_closed.extend(
						node.position for node in closed_list
					)

			return tracer

		return None

	previous_tracer = sys.settrace(tracer)

	try:
		path = a_star(maze, start, end)
	finally:
		sys.settrace(previous_tracer)

	return path, captured_closed


# ============================================================
# ЭВРИСТИКА ДЛЯ A*
# ============================================================
# Адаптирована под Node (как того требует оригинальная a_star
# ниже: calculate_h(child, end_node) вызывается с объектами
# Node, а не с кортежами координат).

def calculate_h(node, goal):
	return abs(node.position[0] - goal.position[0]) + \
		abs(node.position[1] - goal.position[1])


# ============================================================
# A* (ORIGINAL)
# ============================================================

def a_star(maze, start, end):
	start_node = Node(None, start)
	end_node = Node(None, end)

	open_list = []			# O(n)
	closed_list = []

	open_list.append(start_node)

	while len(open_list) > 0:
		current_node = open_list[0]
		current_index = 0

		for index, item in enumerate(open_list):
			if item.f < current_node.f:
				current_node = item
				current_index = index

		open_list.pop(current_index)
		closed_list.append(current_node)

		if current_node == end_node:
			path = []
			current = current_node

			while current is not None:
				path.append(current.position)
				current = current.parent

			return path[::-1]

		children = [
			Node(current_node, (current_node.position[0] - 1, current_node.position[1])),
			Node(current_node, (current_node.position[0] + 1, current_node.position[1])),
			Node(current_node, (current_node.position[0], current_node.position[1] - 1)),
			Node(current_node, (current_node.position[0], current_node.position[1] + 1))
		]

		for child in children:
			if child in closed_list:
				continue

			row, col = child.position

			if not (0 <= row < len(maze) and 0 <= col < len(maze[0])):
				continue

			if maze[row][col] != 0:
				continue

			child.g = current_node.g + 1
			child.h = calculate_h(child, end_node)
			child.f = child.g + child.h

			if len([open_node for open_node in open_list if child == open_node and child.g >
					open_node.g]) > 0:
				continue

			open_list.append(child)

	# Путь не найден
	return []


# ============================================================
# ВИЗУАЛИЗАЦИЯ ИСХОДНОГО ЛАБИРИНТА (добавлено)
# ============================================================

def draw_maze(maze, path=None):
	plt.figure(figsize=(9, 9))

	plt.imshow(
		maze,
		cmap='binary',
		interpolation='nearest'
	)

	if path:
		path_rows = [point[0] for point in path]
		path_cols = [point[1] for point in path]

		plt.plot(
			path_cols,
			path_rows,
			linewidth=2,
			label='Путь'
		)

	plt.scatter(
		0,
		0,
		s=100,
		label='Старт'
	)

	plt.scatter(
		len(maze[0]) - 1,
		len(maze) - 1,
		s=100,
		marker='x',
		label='Цель'
	)

	plt.title('Лабиринт')
	plt.legend()
	plt.grid(False)
	plt.show()


# ============================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТА ПОИСКА (добавлено)
# ============================================================

def draw_path(maze, path, visited, title):
	plt.figure(figsize=(9, 9))

	plt.imshow(
		maze,
		cmap='binary',
		interpolation='nearest'
	)

	if visited:
		visited_rows = [point[0] for point in visited]
		visited_cols = [point[1] for point in visited]

		plt.scatter(
			visited_cols,
			visited_rows,
			s=8,
			alpha=0.35,
			label='Посещённые клетки'
		)

	if path:
		path_rows = [point[0] for point in path]
		path_cols = [point[1] for point in path]

		plt.plot(
			path_cols,
			path_rows,
			linewidth=2,
			label='Найденный путь'
		)

	plt.scatter(
		0,
		0,
		s=120,
		label='Старт',
		zorder=5
	)

	plt.scatter(
		len(maze[0]) - 1,
		len(maze) - 1,
		s=120,
		marker='X',
		label='Цель',
		zorder=5
	)

	plt.title(title)
	plt.legend()
	plt.grid(False)
	plt.show()


# ============================================================
# СРАВНЕНИЕ АЛГОРИТМОВ (добавлено)
# ============================================================

def draw_comparison(
	maze,
	dijkstra_path,
	dijkstra_visited,
	a_star_path,
	a_star_visited
):
	fig, axes = plt.subplots(
		1,
		2,
		figsize=(16, 8)
	)

	# --------------------------------------------------------
	# ДЕЙКСТРА
	# --------------------------------------------------------

	axes[0].imshow(
		maze,
		cmap='binary',
		interpolation='nearest'
	)

	if dijkstra_visited:
		rows = [point[0] for point in dijkstra_visited]
		cols = [point[1] for point in dijkstra_visited]

		axes[0].scatter(
			cols,
			rows,
			s=8,
			alpha=0.35,
			label='Посещённые клетки'
		)

	if dijkstra_path:
		rows = [point[0] for point in dijkstra_path]
		cols = [point[1] for point in dijkstra_path]

		axes[0].plot(
			cols,
			rows,
			linewidth=2,
			label='Путь'
		)

	axes[0].scatter(0, 0, s=120, label='Старт', zorder=5)
	axes[0].scatter(COLS - 1, ROWS - 1, s=120, marker='X', label='Цель', zorder=5)
	axes[0].set_title('Алгоритм Дейкстры')
	axes[0].legend()

	# --------------------------------------------------------
	# A*
	# --------------------------------------------------------

	axes[1].imshow(
		maze,
		cmap='binary',
		interpolation='nearest'
	)

	if a_star_visited:
		rows = [point[0] for point in a_star_visited]
		cols = [point[1] for point in a_star_visited]

		axes[1].scatter(
			cols,
			rows,
			s=8,
			alpha=0.35,
			label='Посещённые клетки'
		)

	if a_star_path:
		rows = [point[0] for point in a_star_path]
		cols = [point[1] for point in a_star_path]

		axes[1].plot(
			cols,
			rows,
			linewidth=2,
			label='Путь'
		)

	axes[1].scatter(0, 0, s=120, label='Старт', zorder=5)
	axes[1].scatter(COLS - 1, ROWS - 1, s=120, marker='X', label='Цель', zorder=5)
	axes[1].set_title('Алгоритм A*')
	axes[1].legend()

	plt.suptitle('Сравнение алгоритмов поиска пути', fontsize=16)
	plt.tight_layout()
	plt.show()


# ============================================================
# ВИЗУАЛИЗАЦИЯ ГРАФА (добавлено)
# ============================================================

def draw_graph(graph):
	G = nx.Graph()

	for vertex in graph:
		for neighbor, weight in graph[vertex].items():
			G.add_edge(vertex, neighbor, weight=weight)

	position = nx.spring_layout(G, seed=42)

	plt.figure(figsize=(8, 6))

	nx.draw(
		G,
		position,
		with_labels=True,
		node_size=1000
	)

	labels = nx.get_edge_attributes(G, 'weight')

	nx.draw_networkx_edge_labels(
		G,
		position,
		edge_labels=labels
	)

	plt.title('Взвешенный граф')
	plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

	# --------------------------------------------------------
	# ЧАСТЬ 1. ВЗВЕШЕННЫЙ ГРАФ
	# --------------------------------------------------------
	# build_graph игнорирует edges (как в оригинале), поэтому
	# рёбра добавляются отдельно через populate_graph — сам
	# build_graph не меняется.

	edges = [
		(0, 1, 4),
		(0, 2, 2),
		(1, 2, 1),
		(1, 3, 5),
		(2, 3, 8),
		(2, 4, 10),
		(3, 4, 2)
	]

	graph = build_graph(edges, 5)
	graph = populate_graph(graph, edges)

	print('Расстояния от вершины 0:')
	print(dijkstra(graph, 0))

	draw_graph(graph)

	# Кратчайшие расстояния между всеми городами — алгоритмом
	# Дейкстры и алгоритмом A* (см. a_star_graph выше). Оба
	# набора расстояний должны совпадать, так как A* с
	# эвристикой h=0 корректен и по сути повторяет Дейкстру.

	print()
	print('Кратчайшие расстояния между всеми городами:')

	dijkstra_distances = all_distances_dijkstra(graph)
	a_star_distances = all_distances_a_star(graph)

	print('Алгоритм Дейкстры:')
	for city, distances in dijkstra_distances.items():
		print(f'  Город {city}: {distances}')

	print('Алгоритм A*:')
	for city, distances in a_star_distances.items():
		print(f'  Город {city}: {distances}')

	# Достижимость в пределах mileage и город с минимальным
	# количеством достижимых городов.

	mileage = 6

	dijkstra_reachable = count_reachable_within_mileage(
		dijkstra_distances,
		mileage
	)

	a_star_reachable = count_reachable_within_mileage(
		a_star_distances,
		mileage
	)

	min_city_dijkstra = find_city_with_min_reachable(dijkstra_reachable)
	min_city_a_star = find_city_with_min_reachable(a_star_reachable)

	print()
	print(f'Значение mileage: {mileage}')

	print('Количество достижимых городов (Дейкстра):', dijkstra_reachable)
	print(
		'Город с минимальным числом достижимых городов (Дейкстра):',
		min_city_dijkstra,
		'->',
		dijkstra_reachable[min_city_dijkstra]
	)

	print('Количество достижимых городов (A*):', a_star_reachable)
	print(
		'Город с минимальным числом достижимых городов (A*):',
		min_city_a_star,
		'->',
		a_star_reachable[min_city_a_star]
	)

	# --------------------------------------------------------
	# ЧАСТЬ 2. ГЕНЕРАЦИЯ ЛАБИРИНТА
	# --------------------------------------------------------
	# generate_maze сама по себе не гарантирует существование пути
	# (как в оригинале) и не меняется. Вместо этого лабиринт
	# перегенерируется через ensure_solvable_maze, пока старт и
	# финиш не окажутся связаны проходом.

	print()
	print('Генерация лабиринта...')

	maze = ensure_solvable_maze(ROWS, COLS, wall_prob=0.3)

	print('Лабиринт создан (гарантированно проходим).')

	draw_maze(maze)

	start = (0, 0)
	goal = (ROWS - 1, COLS - 1)

	# --------------------------------------------------------
	# ДЕЙКСТРА
	# --------------------------------------------------------

	print()
	print('Запуск алгоритма Дейкстры...')

	start_time = time.perf_counter()
	dijkstra_path, dijkstra_visited = dijkstra_maze(maze, start, goal)
	dijkstra_time = time.perf_counter() - start_time

	print('Дейкстра завершён.')

	if dijkstra_path:
		print('Длина пути:', len(dijkstra_path) - 1)
	else:
		print('Путь не найден.')

	print('Посещено вершин:', len(dijkstra_visited))
	print('Время выполнения:', f'{dijkstra_time:.6f} с')

	draw_path(
		maze,
		dijkstra_path,
		dijkstra_visited,
		'Поиск пути алгоритмом Дейкстры'
	)

	# --------------------------------------------------------
	# A*
	# --------------------------------------------------------

	print()
	print('Запуск алгоритма A*...')

	a_star_maze = build_tree_maze(maze, start)
	start_expanded = to_expanded_position(start)
	goal_expanded = to_expanded_position(goal)

	start_time = time.perf_counter()
	a_star_path_expanded = a_star(a_star_maze, start_expanded, goal_expanded)
	a_star_time = time.perf_counter() - start_time

	# Отдельный, не таймируемый прогон — только для отрисовки
	# посещённых клеток, время этого вызова никуда не идёт.
	_, a_star_closed_expanded = run_a_star_with_visited(
		a_star_maze,
		start_expanded,
		goal_expanded
	)

	a_star_path = collapse_expanded_path(a_star_path_expanded)

	a_star_visited = {
		(row // 2, col // 2)
		for row, col in a_star_closed_expanded
		if row % 2 == 0 and col % 2 == 0
	}

	print('A* завершён.')

	if a_star_path:
		print('Длина пути:', len(a_star_path) - 1)
	else:
		print('Путь не найден.')

	print('Посещено вершин:', len(a_star_visited))
	print('Время выполнения:', f'{a_star_time:.6f} с')

	draw_path(
		maze,
		a_star_path,
		a_star_visited,
		'Поиск пути алгоритмом A*'
	)

	# --------------------------------------------------------
	# ИТОГОВОЕ СРАВНЕНИЕ
	# --------------------------------------------------------

	print()
	print('Сравнение алгоритмов:')

	print(
		f'Дейкстра: путь = {len(dijkstra_path) - 1 if dijkstra_path else "нет"}, '
		f'посещено = {len(dijkstra_visited)}, '
		f'время = {dijkstra_time:.6f} с'
	)

	print(
		f'A*: путь = {len(a_star_path) - 1 if a_star_path else "нет"}, '
		f'посещено = {len(a_star_visited)}, '
		f'время = {a_star_time:.6f} с'
	)

	draw_comparison(
		maze,
		dijkstra_path,
		dijkstra_visited,
		a_star_path,
		a_star_visited
	)


if __name__ == '__main__':
	main()