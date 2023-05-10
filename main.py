import random
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

class Obstacle:
    def __init__(self, x, y, z, size, obstacle_type):
        self.x = x
        self.y = y
        self.z = z
        self.size = size
        self.type = obstacle_type

def generate_obstacles(width, height, depth, num_obstacles):
    obstacles = []
    obstacle_types = ['tree', 'tractor', 'haystack']

    for _ in range(num_obstacles):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        obstacle_type = random.choice(obstacle_types)
        size = random.uniform(1, 4)
        
        if obstacle_type == 'tree':
            z = random.uniform(1, depth - 1)
        else:
            z = 0
        
        obstacle = Obstacle(x, y, z, size, obstacle_type)
        obstacles.append(obstacle)

    return obstacles

class GridMap3D:
    def __init__(self, width, height, depth, obstacles):
        self.width = width
        self.height = height
        self.depth = depth
        self.obstacles = obstacles

    def in_bounds(self, node):
        (x, y, z) = node
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def passable(self, node):
        for obstacle in self.obstacles:
            if (obstacle.x - obstacle.size / 2 <= node[0] <= obstacle.x + obstacle.size / 2) and \
               (obstacle.y - obstacle.size / 2 <= node[1] <= obstacle.y + obstacle.size / 2) and \
               (obstacle.z - obstacle.size / 2 <= node[2] <= obstacle.z + obstacle.size / 2):
                return False
        return True

    def neighbors(self, node):
        (x, y, z) = node
        neighbors = [
            (x + dx, y + dy, z + dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if dx != 0 or dy != 0 or dz != 0
        ]
        neighbors = filter(self.in_bounds, neighbors)
        neighbors = filter(self.passable, neighbors)
        return neighbors

    def cost(self, current, next):
        return np.sqrt((next[0] - current[0]) ** 2 + (next[1] - current[1]) ** 2 + (next[2] - current[2]) ** 2)

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def a_star_search(graph, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def plot_cylinder(ax, base_center, height, radius, color):
    x, y, z = base_center
    resolution = 20
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, height, resolution)
    PHI, THETA = np.meshgrid(phi, theta)

    X = x + radius * np.cos(PHI)
    Y = y + radius * np.sin(PHI)
    Z = z + THETA

    ax.plot_surface(X, Y, Z, color=color, shade=True, alpha=0.8)

def plot_cube(ax, center, size, color):
    x, y, z = center
    x -= size / 2
    y -= size / 2
    z -= size / 2

    vertices = np.array([
        [x, y, z],
        [x + size, y, z],
        [x + size, y + size, z],
        [x, y + size, z],
        [x, y, z + size],
        [x + size, y, z + size],
        [x + size, y + size, z + size],
        [x, y + size, z + size]
    ])

    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[7], vertices[6], vertices[5], vertices[4]],
        [vertices[0], vertices[4], vertices[7], vertices[3]],
        [vertices[1], vertices[5], vertices[6], vertices[2]]
    ]

    face_collection = Poly3DCollection(faces, edgecolors='k', linewidths=1, alpha=0.8)
    face_collection.set_facecolor(color)
    ax.add_collection3d(face_collection)

def plot_obstacle(ax, obstacle):
    if obstacle.type == 'tree':
        plot_cylinder(ax, (obstacle.x, obstacle.y, 0), obstacle.z, obstacle.size / 2, 'green')
    elif obstacle.type == 'tractor' or obstacle.type == 'haystack':
        plot_cube(ax, (obstacle.x, obstacle.y, obstacle.z), obstacle.size, 'red')

def plot_drone(ax, position, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = position[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = position[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = position[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='cyan', shade=True, alpha=0.8)

def run_multiple_drones(grid_map, drone_starts, drone_goals):
    paths = []
    for start, goal in zip(drone_starts, drone_goals):
        came_from, cost_so_far = a_star_search(grid_map, start, goal)
        path = reconstruct_path(came_from, start, goal)
        paths.append(path)
    return paths

def plot_path_on_map_3d(grid_map, paths, drone_starts, drone_goals):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for obstacle in grid_map.obstacles:
        plot_obstacle(ax, obstacle)

    for path in paths:
        xs = [node[0] for node in path]
        ys = [node[1] for node in path]
        zs = [node[2] for node in path]

        ax.plot(xs, ys, zs, linewidth=2, color='blue', marker='o', markersize=3)

    for start, goal in zip(drone_starts, drone_goals):
        plot_drone(ax, start, 1)
        plot_drone(ax, goal, 1)

    plt.show()

def update_obstacles(obstacles):
    # 在这个示例中，我们随机移动障碍物
    # 您可以根据需要修改这个函数以实现特定的障碍物更新逻辑
    for obstacle in obstacles:
        obstacle.x += random.randint(-1, 1)
        obstacle.y += random.randint(-1, 1)
        obstacle.z += random.randint(-1, 1)
    return obstacles

def update_obstacles(obstacles, width, height, depth):
    for obstacle in obstacles:
        obstacle.x += random.randint(-1, 1)
        obstacle.y += random.randint(-1, 1)
        obstacle.z += random.randint(-1, 1)

        # 确保障碍物坐标在地图范围内
        obstacle.x = max(0, min(obstacle.x, width - 1))
        obstacle.y = max(0, min(obstacle.y, height - 1))
        obstacle.z = max(0, min(obstacle.z, depth - 1))

    return obstacles

# def update_frame(frame):
#     global obstacles, grid_map, drone_start, drone_goal

#     width, height, depth = 50, 50, 10
#     num_obstacles = random.randint(10, 60)
#     obstacles = generate_obstacles(width, height, depth, num_obstacles)
#     obstacles = update_obstacles(obstacles, width, height, depth)
#     grid_map = GridMap3D(width, height, depth, obstacles)

#     # 重新计算路径
#     came_from, cost_so_far = a_star_search(grid_map, drone_start, drone_goal)
#     path = reconstruct_path(came_from, drone_start, drone_goal)

#     ax.clear()
#     for obstacle in grid_map.obstacles:
#         plot_obstacle(ax, obstacle)

#     xs = [node[0] for node in path]
#     ys = [node[1] for node in path]
#     zs = [node[2] for node in path]

#     ax.plot(xs, ys, zs, linewidth=2, color='blue', marker='o', markersize=3)

#     plot_drone(ax, drone_start, 1)
#     plot_drone(ax, drone_goal, 1)

def update_frame(frame):
    global obstacles, grid_map, drone_start, drone_goal

    width, height, depth = 50, 50, 10
    num_obstacles = random.randint(1, 2)
    obstacles = generate_obstacles(width, height, depth, num_obstacles)
    obstacles = update_obstacles(obstacles, width, height, depth)
    grid_map = GridMap3D(width, height, depth, obstacles)

    # 计算螺旋路径
    spiral_path_points = spiral_path(drone_start, drone_goal, 100, 3)

    # 计算升空和降落路径
    takeoff_point = (drone_start[0], drone_start[1], spiral_path_points[0][2])
    landing_point = (drone_goal[0], drone_goal[1], spiral_path_points[-1][2])
    takeoff_path = vertical_transition(drone_start, takeoff_point, 10)
    landing_path = vertical_transition(landing_point, drone_goal, 10)

    # 将整个路径拼接在一起
    full_path = takeoff_path + spiral_path_points + landing_path

    # 将路径点添加到网格地图
    for point in full_path:
        grid_map.obstacles.append(Obstacle(point[0], point[1], point[2], 0.5, 'path'))

    # 重新计算路径
    came_from, cost_so_far = a_star_search(grid_map, drone_start, drone_goal)
    path = reconstruct_path(came_from, drone_start, drone_goal)

    ax.clear()
    for obstacle in grid_map.obstacles:
        if obstacle.type != 'path':
            plot_obstacle(ax, obstacle)

    # 绘制完整路径
    full_xs = [node[0] for node in full_path]
    full_ys = [node[1] for node in full_path]
    full_zs = [node[2] for node in full_path]
    ax.plot(full_xs, full_ys, full_zs, linewidth=2, color='magenta', marker='o', markersize=3)

    # 绘制避障后的路径
    xs = [node[0] for node in path]
    ys = [node[1] for node in path]
    zs = [node[2] for node in path]
    ax.plot(xs, ys, zs, linewidth=2, color='blue', marker='o', markersize=3)

    plot_drone(ax, drone_start, 1)
    plot_drone(ax, drone_goal, 1)

#螺旋路径算法
def spiral_path(start, goal, num_points, num_turns):
    x_start, y_start, z_start = start
    x_goal, y_goal, z_goal = goal
    
    t = np.linspace(0, 1, num_points)
    x_spiral = x_start + (x_goal - x_start) * t * np.cos(2 * np.pi * num_turns * t)
    y_spiral = y_start + (y_goal - y_start) * t
    z_spiral = z_start + (z_goal - z_start) * t * np.sin(2 * np.pi * num_turns * t)

    return list(zip(x_spiral, y_spiral, z_spiral))

def vertical_transition(start, goal, num_points):
    x_start, y_start, z_start = start
    x_goal, y_goal, z_goal = goal
    
    t = np.linspace(0, 1, num_points)
    x_transition = np.full(num_points, x_start + (x_goal - x_start) * t[-1])
    y_transition = np.full(num_points, y_start + (y_goal - y_start) * t[-1])
    z_transition = z_start + (z_goal - z_start) * t

    return list(zip(x_transition, y_transition, z_transition))


# 定义无人机的起点和终点
drone_start = (0, 0, 0)
drone_goal = (49, 49, 30)

# 设定仿真的时间步数
num_timesteps = 10

# Set up the figure and axis for the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the animation
ani = animation.FuncAnimation(fig, update_frame, frames=range(num_timesteps), interval=1000, blit=False)

plt.show()