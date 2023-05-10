import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree

# from pyplotz.pyplotz import PyplotZ
# pltz = PyplotZ()
# pltz.enable_chinese()
random.seed(0)

# Function to create a 3D box
def create_box(x, y, z, dx, dy, dz):
    return [
        [(x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z)],
        [(x, y, z), (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y, z)],
        [(x, y, z), (x, y + dy, z), (x, y + dy, z + dz), (x, y, z + dz)],
        [
            (x + dx, y, z),
            (x + dx, y + dy, z),
            (x + dx, y + dy, z + dz),
            (x + dx, y, z + dz),
        ],
        [
            (x, y + dy, z),
            (x, y + dy, z + dz),
            (x + dx, y + dy, z + dz),
            (x + dx, y + dy, z),
        ],
        [
            (x, y, z + dz),
            (x + dx, y, z + dz),
            (x + dx, y + dy, z + dz),
            (x, y + dy, z + dz),
        ],
    ]


# TreeNode class
class TreeNode:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent


# Check if point is in obstacle
def point_in_obstacle(point, obstacles):
    for obstacle in obstacles:
        if (
            obstacle[0][0] <= point[0] <= (obstacle[0][0] + obstacle[1][0])
            and obstacle[0][1] <= point[1] <= (obstacle[0][1] + obstacle[1][1])
            and obstacle[0][2] <= point[2] <= (obstacle[0][2] + obstacle[1][2])
        ):
            return True
    return False


# Generate a random point
def generate_random_point(x_max, y_max, z_min, z_max):
    return np.array(
        [
            random.uniform(0, x_max),
            random.uniform(0, y_max),
            random.uniform(z_min, z_max),
        ]
    )


# RRT algorithm
def create_ground_grid(x_max, y_max, grid_size):
    grid = []
    for x in range(0, x_max, grid_size):
        for y in range(0, y_max, grid_size):
            grid.append((x + grid_size // 2, y + grid_size // 2))
    return np.array(grid)


# RRT algorithm
def rrt(start, goal, obstacles, num_iterations=20, max_distance=200):
    x_max, y_max = 1000, 1000
    z_min, z_max = 100, 300
    # spray_height = 150

    start_node = TreeNode(start)
    goal_node = TreeNode(goal)

    tree_nodes = [start_node]
    tree_kdtree = KDTree(start.reshape(1, -1))

    ground_grid = create_ground_grid(x_max, y_max, 80)

    for _ in range(num_iterations):
        random_point = generate_random_point(x_max, y_max, z_min, z_max)

        if point_in_obstacle(random_point, obstacles):
            continue

        nearest_index = tree_kdtree.query(
            random_point.reshape(1, -1), return_distance=False
        )[0][0]
        nearest_node = tree_nodes[nearest_index]

        new_point = (
            nearest_node.point
            + (random_point - nearest_node.point)
            / np.linalg.norm(random_point - nearest_node.point)
            * max_distance
        )
        # new_point[2] = spray_height  # Set the height to the spraying height

        if point_in_obstacle(new_point, obstacles):
            continue

        new_node = TreeNode(new_point, parent=nearest_node)
        tree_nodes.append(new_node)
        tree_kdtree = KDTree(np.vstack([tree_kdtree.data, new_point.reshape(1, -1)]))

        if np.linalg.norm(new_point - goal) <= max_distance:
            goal_node.parent = new_node
            break

    path = []
    current_node = goal_node

    while current_node.parent is not None:
        path.append(current_node.point)
        current_node = current_node.parent

    path.append(start)
    path.reverse()

    return path

# Check if a grid cell is obstructed by an obstacle
def is_obstructed(cell, obstacles, grid_size):
    for obstacle in obstacles:
        x_min, y_min, z_min = obstacle[0]
        dx, dy, dz = obstacle[1]
        x_max, y_max, z_max = x_min + dx, y_min + dy, z_min + dz

        if x_min <= cell[0] <= x_max and y_min <= cell[1] <= y_max:
            return True
    return False


# Create ground grid, excluding obstructed cells
def create_ground_grid(x_max, y_max, grid_size, obstacles):
    grid = []
    for x in range(0, x_max, grid_size):
        for y in range(0, y_max, grid_size):
            cell = (x + grid_size // 2, y + grid_size // 2)
            if not is_obstructed(cell, obstacles, grid_size):
                grid.append(cell)
    return np.array(grid)


# Greedy traversal of grid cells
def greedy_traversal(start, goal, ground_grid):
    unvisited = set(range(len(ground_grid)))
    path = [start]
    current_index = None

    # Find the closest start cell
    min_distance = float("inf")
    for i, cell in enumerate(ground_grid):
        distance = np.linalg.norm(start[:2] - np.array([*cell]))
        if distance < min_distance:
            min_distance = distance
            current_index = i

    unvisited.remove(current_index)
    start_z = start[2]
    goal_z = goal[2]

    # Compute the next point's z value based on the relative distance from the start to the goal
    next_z = start_z + (1 - (min_distance / np.linalg.norm(goal[:2] - start[:2]))) * (goal_z - start_z)
    path.append(np.array([*ground_grid[current_index], next_z]))

    while unvisited:
        min_distance = float("inf")
        next_index = None

        for i in unvisited:
            distance = np.linalg.norm(path[-1][:2] - np.array([*ground_grid[i]]))
            if distance < min_distance:
                min_distance = distance
                next_index = i

        unvisited.remove(next_index)
        current_index = next_index

        # Compute the next point's z value based on the relative distance from the start to the goal
        next_z = start_z + (1 - (min_distance / np.linalg.norm(goal[:2] - start[:2]))) * (goal_z - start_z)
        path.append(np.array([*ground_grid[current_index], next_z]))

    return path

def gradient_descent_path_smooth(
    path, obstacles_info, alpha=0.01, beta=0.08, max_iterations=15, tolerance=1e-5
):
    path = path.astype(np.float64)

    def path_cost(path, obstacles_info, beta):
        cost = 0
        for i in range(1, len(path) - 1):
            cost += np.linalg.norm(
                path[i - 1] - 2 * path[i] + path[i + 1]
            )  # Smoothness cost
            for obs in obstacles_info:
                min_dist_to_obstacle = np.min(np.linalg.norm(path[i][:2] - obs[0][:2]))
                if min_dist_to_obstacle < obs[1][0]:
                    cost += beta * (
                        1 - min_dist_to_obstacle / obs[1][0]
                    )  # Obstacle cost
        return cost

    def gradient(path, i, obstacles, beta):
        grad = np.zeros(3)
        grad += (
            4 * path[i] - 2 * path[i - 1] - 2 * path[i + 1]
        )  # Smoothness gradient

        for obstacle in obstacles:
            x_min, y_min, z_min = obstacle[0]
            dx, dy, dz = obstacle[1]
            x_max, y_max, z_max = x_min + dx, y_min + dy, z_min + dz
            obs_center = np.array([x_min + dx / 2, y_min + dy / 2, z_min + dz / 2])
            min_dist_to_obstacle = np.linalg.norm(path[i] - obs_center)

            if (
                x_min <= path[i][0] <= x_max
                and y_min <= path[i][1] <= y_max
                and z_min <= path[i][2] <= z_max
            ):
                grad[:2] += beta * (
                    -2 * (path[i][:2] - obs_center[:2]) / min_dist_to_obstacle
                )  # Obstacle gradient

        return grad

    optimized_path = path.copy()
    for _ in range(max_iterations):
        new_path = optimized_path.copy()
        for i in range(1, len(optimized_path) - 1):
            new_path[i] -= alpha * gradient(optimized_path, i, obstacles_info, beta)
        if np.linalg.norm(new_path - optimized_path) < tolerance:
            break
        optimized_path = new_path

    return optimized_path

# Scene setup
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Ground
X, Y = np.meshgrid(np.linspace(0, 1000, 2), np.linspace(0, 1000, 2))
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, alpha=0.1, color="green")


def generate_obstacles(num_obstacles, x_lim, y_lim, min_distance, start, goal):
    obstacles_info = []
    for _ in range(num_obstacles):
        while True:
            x, y = np.random.randint(0, x_lim - 100, size=2)
            z = 0
            dx, dy = np.random.randint(80, 121, size=2)  # 底面积在 60x60 到 100x100 之间
            dz = np.random.randint(100, 351)  # 高度在 50 到 350 之间
            new_obstacle = ((x, y, z), (dx, dy, dz))

            # 检查新障碍物与现有障碍物之间的距离
            min_distance_to_others = float("inf")
            for obstacle in obstacles_info:
                center1 = np.array([x + dx / 2, y + dy / 2, z + dz / 2])
                center2 = np.array(
                    [
                        obstacle[0][0] + obstacle[1][0] / 2,
                        obstacle[0][1] + obstacle[1][1] / 2,
                        obstacle[0][2] + obstacle[1][2] / 2,
                    ]
                )
                min_distance_to_others = min(
                    min_distance_to_others, np.linalg.norm(center1 - center2)
                )

            # 如果新障碍物与现有障碍物之间的距离大于最小距离且起点和终点不在障碍物中，则添加到障碍物列表中
            if min_distance_to_others >= min_distance and not (
                point_in_obstacle(start, [new_obstacle])
                or point_in_obstacle(goal, [new_obstacle])
            ):
                obstacles_info.append(new_obstacle)
                break

    return obstacles_info

num_obstacles = 15
x_lim, y_lim = 950, 950
min_distance = 150

start = np.array([50, 50, 100])
goal = np.array([900, 900, 300])

obstacles_info = generate_obstacles(
    num_obstacles, x_lim, y_lim, min_distance, start, goal
)


for obstacle_info in obstacles_info:
    obstacle = create_box(*obstacle_info[0], *obstacle_info[1])
    face_colors = np.random.rand(3)
    ax.add_collection3d(
        Poly3DCollection(
            obstacle, facecolors=face_colors, linewidths=1, edgecolors="k", alpha=0.7
        )
    )

# Obstacles
# num_obstacles = 20
# np.random.seed(42)

# obstacles_info = []
# for _ in range(num_obstacles):
#     x, y = np.random.randint(0, 950, size=2)
#     z = 0
#     dx, dy = np.random.randint(50, 80, size=2)
#     dz = np.random.randint(120, 300)
#     obstacle = create_box(x, y, z, dx, dy, dz)
#     obstacles_info.append(((x, y, z), (dx, dy, dz)))
#     face_colors = np.random.rand(3)
#     ax.add_collection3d(
#         Poly3DCollection(
#             obstacle, facecolors=face_colors, linewidths=1, edgecolors="k", alpha=0.7
#         )
#     )


for obstacle in obstacles_info:
    box = Poly3DCollection(
        create_box(*obstacle[0], *obstacle[1]), alpha=0.5, linewidths=1, edgecolors="k"
    )
    box.set_facecolor("gray")
    ax.add_collection3d(box)


# Create ground grid
ground_grid = create_ground_grid(1000, 1000, 50, obstacles_info)

# Greedy traversal
# Greedy traversal
path = greedy_traversal(start, goal, ground_grid)

path = np.array(path + [goal])
start_height = start[2]
goal_height = goal[2]
heights = np.linspace(start_height, goal_height, len(path))
path[:, 2] = heights  # Set the height to the spraying height

ax.plot(path[:, 0], path[:, 1], path[:, 2], "r-")
ax.scatter(path[:, 0], path[:, 1], path[:, 2], c="red", s=50)

# Plot start and goal points
ax.scatter(*start, c="blue", s=100, marker="o")
ax.scatter(*goal, c="blue", s=100, marker="x")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")




obstacle_colors = [
    "#FFA07A",
    "#7FFF00",
    "#D2691E",
    "#DC143C",
    "#008B8B",
    "#ADFF2F",
    "#8B008B",
    "#FF8C00",
    "#FF1493",
    "#FFD700",
]

# Add obstacles to the 2D plot
for obstacle, color in zip(obstacles_info, obstacle_colors):
    x_min, y_min, z_min = obstacle[0]
    dx, dy, dz = obstacle[1]
    rect = plt.Rectangle(
        (x_min, y_min), dx, dy, facecolor=color, edgecolor="black", alpha=0.5
    )
    plt.gca().add_patch(rect)


# Smooth and plot optimized 2D path
smoothed_path = np.array(gradient_descent_path_smooth(path, obstacles_info))

plt.close("all")
fig = plt.figure(figsize=(20, 10))

# 3D Path
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(path[:, 0], path[:, 1], path[:, 2], "r-", linewidth=3, label="Optimized 3D Path")
ax1.scatter(*start, c="blue", s=100, marker="o", label="Start")
ax1.scatter(*goal, c="blue", s=100, marker="x", label="Goal")
ax1.view_init(elev=40, azim=215)

# 2D Path
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(smoothed_path[:, 0], smoothed_path[:, 1], "g-", label="Optimized 2D Path")
ax2.scatter(smoothed_path[:, 0], smoothed_path[:, 1], c="green", s=50)
ax2.scatter(*start[:2], c="blue", s=100, marker="o", label="Start")
ax2.scatter(*goal[:2], c="blue", s=100, marker="x", label="Goal")

# Add obstacles to both plots
for i, obstacle in enumerate(obstacles_info):
    color = obstacle_colors[i % len(obstacle_colors)]
    x_min, y_min, z_min = obstacle[0]
    dx, dy, dz = obstacle[1]

    # 3D plot
    ax1.bar3d(x_min, y_min, z_min, dx, dy, dz, color=color, shade=True, alpha=0.5)

    # 2D plot
    rect = plt.Rectangle(
        (x_min, y_min), dx, dy, facecolor=color, edgecolor="black", alpha=0.5
    )
    ax2.add_patch(rect)

# Set labels and titles
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_title("Optimized 3D Path")

ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Optimized 2D Path")
ax2.legend()
ax2.axis("equal")
ax2.grid(True)

# Show the plots
plt.show()