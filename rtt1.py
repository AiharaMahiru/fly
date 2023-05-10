import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from sklearn.neighbors import KDTree

# Function to create a 3D box
def create_box(x, y, z, dx, dy, dz):
    return [
        [(x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z)],
        [(x, y, z), (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y, z)],
        [(x, y, z), (x, y + dy, z), (x, y + dy, z + dz), (x, y, z + dz)],
        [(x + dx, y, z), (x + dx, y + dy, z), (x + dx, y + dy, z + dz), (x + dx, y, z + dz)],
        [(x, y + dy, z), (x, y + dy, z + dz), (x + dx, y + dy, z + dz), (x + dx, y + dy, z)],
        [(x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)],
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
def generate_random_point(x_max, y_max, z_max):
    return np.array([random.uniform(0, x_max), random.uniform(0, y_max), random.uniform(0, z_max)])

# RRT algorithm
def rrt(start, goal, obstacles, num_iterations=10000, max_distance=50):
    x_max, y_max, z_max = 1000, 1000, 500

    start_node = TreeNode(start)
    goal_node = TreeNode(goal)

    tree_nodes = [start_node]
    tree_kdtree = KDTree(start.reshape(1, -1))

    for _ in range(num_iterations):
        random_point = generate_random_point(x_max, y_max, z_max)

        if point_in_obstacle(random_point, obstacles):
            continue

        nearest_index = tree_kdtree.query(random_point.reshape(1, -1), return_distance=False)[0][0]
        nearest_node = tree_nodes[nearest_index]

        new_point = nearest_node.point + (random_point - nearest_node.point) / np.linalg.norm(random_point - nearest_node.point) * max_distance

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

# Scene setup
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Ground
X, Y = np.meshgrid(np.linspace(0, 1000, 2), np.linspace(0, 1000, 2))
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, alpha=0.5)

# Obstacles
num_obstacles = 35
np.random.seed(42)

obstacles_info = []
for _ in range(num_obstacles):
    x, y = np.random.randint(0, 950, size=2)
    z = 0
    dx, dy = np.random.randint(50, 100, size=2)
    dz = np.random.randint(80, 400)
    obstacle = create_box(x, y, z, dx, dy, dz)
    obstacles_info.append(((x, y, z), (dx, dy, dz)))
    face_colors = np.random.rand(3)
    ax.add_collection3d(Poly3DCollection(obstacle, facecolors=face_colors, linewidths=1, edgecolors="k", alpha=0.7))

# Destination point
destination = [990, 990, 10]
ax.scatter(*destination, c="r", marker="o", s=100, label="Destination")

drone = create_box(0, 0, 10, 30, 30, 30)
drone_face_colors = np.array([0, 0, 1])  # Blue
ax.add_collection3d(Poly3DCollection(drone, facecolors=drone_face_colors, linewidths=1, edgecolors="k", alpha=1))

# Axes labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Set axes limits
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 500)

# Use RRT to find the path
start = np.array([15, 15, 25])
goal = np.array(destination)
path = rrt(start, goal, obstacles_info)

# Plot the path
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], path[:, 2], c="m", linewidth=3, label="Path")

plt.legend()
plt.show()

#num_iterations , max_distance = 搜索速度和精度