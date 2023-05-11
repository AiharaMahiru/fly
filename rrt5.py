import numpy as np
import random
import heapq

# 网格大小
GRID_SIZE = 30

# 三维空间大小
SPACE_DIMENSIONS = (1000, 1000, 500)

# 无人机和目标位置
START = (30, 30, 30)
TARGET = (900, 900, 450)

# 障碍物数量
NUM_OBSTACLES = 30

# 障碍物底面积范围
OBSTACLE_BASE_RANGE = (50, 100)

# 障碍物高度范围
OBSTACLE_HEIGHT_RANGE = (100, 300)


def generate_obstacles(num_obstacles, space_dimensions, base_range, height_range, grid_size):
    obstacles = np.zeros(tuple(dim // grid_size for dim in space_dimensions), dtype=bool)

    for i in range(num_obstacles):
        base = random.randint(*base_range) // grid_size
        height = random.randint(*height_range) // grid_size

        x = random.randint(0, obstacles.shape[0] - base)
        y = random.randint(0, obstacles.shape[1] - base)
        z = 0

        obstacles[x:x + base, y:y + base, z:z + height] = True

    return obstacles


def get_neighbors(pos, grid_shape):
    neighbors = []
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        x, y, z = pos[0] + dx, pos[1] + dy, pos[2] + dz
        if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
            neighbors.append((x, y, z))
    return neighbors


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def a_star(obstacles, start, target):
    start = tuple(s // GRID_SIZE for s in start)
    target = tuple(t // GRID_SIZE for t in target)

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == target:
            break

        for next_ in get_neighbors(current, obstacles.shape):
            if obstacles[next_]:
                continue

            new_cost = cost_so_far[current] + 1
            if next_ not in cost_so_far or new_cost < cost_so_far[next_]:
                cost_so_far[next_] = new_cost
                priority = new_cost + heuristic(target, next_)
                heapq.heappush(frontier, (priority, next_))
                came_from[next_] = current

    if current != target:
        return None

    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()

    return [(p[0] * GRID_SIZE, p[1] * GRID_SIZE, p[2] * GRID_SIZE) for p in path]


def main():
    obstacles = generate_obstacles(NUM_OBSTACLES, SPACE_DIMENSIONS, OBSTACLE_BASE_RANGE, OBSTACLE_HEIGHT_RANGE, GRID_SIZE)
    path = a_star(obstacles, START, TARGET)

    if path:
        print("Path found:")
        for pos in path:
            print(pos)
    else:
        print("No path found")


if __name__ == '__main__':
    main()