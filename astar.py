import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree as KDTree
import random
# import numpy as np
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

# Scene setup
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Ground
X, Y = np.meshgrid(np.linspace(0, 1000, 2), np.linspace(0, 1000, 2))
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, alpha=0.5)

# Obstacles
num_obstacles = 24
np.random.seed(42)

for _ in range(num_obstacles):
    x, y = np.random.randint(0, 950, size=2)
    z = 0
    dx, dy = np.random.randint(50, 100, size=2)
    dz = np.random.randint(80, 400)
    obstacle = create_box(x, y, z, dx, dy, dz)
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

# Show plot
# plt.show()

def is_valid_move(start, end, obstacles_kdtree, drone_size):
    if any(coord < 0 or coord > 1000 for coord in end) or end[2] < 0 or end[2] > 500:
        return False

    line = np.linspace(start, end, num=2 * int(np.linalg.norm(np.array(start) - np.array(end))))
    distances, _ = obstacles_kdtree.query(line)
    return np.all(distances >= drone_size)

# RRT algorithm
def rrt(start, goal, obstacles_kdtree, drone_size, max_iterations=1000, goal_probability=0.1):
    nodes = [start]
    path = None

    for _ in range(max_iterations):
        if random.random() < goal_probability:
            target = goal
        else:
            target = random_state()

        nearest = nearest_neighbor(target, nodes)
        new_node = steer(nearest, target, drone_size)

        if is_valid_move(nearest, new_node, obstacles_kdtree, drone_size):
            nodes.append(new_node)

            if np.linalg.norm(np.array(new_node) - np.array(goal)) <= drone_size:
                path = build_path(nodes, len(nodes) - 1)
                break

    return nodes, path

# Helper functions for RRT
def random_state():
    x = random.uniform(0, 1000)
    y = random.uniform(0, 1000)
    z = random.uniform(50, 300)
    return (x, y, z)

def nearest_neighbor(target, nodes):
    return min(nodes, key=lambda node: np.linalg.norm(np.array(node) - np.array(target)))

def steer(source, target, step_size):
    direction = np.array(target) - np.array(source)
    direction = step_size * direction / np.linalg.norm(direction)
    new_node = np.array(source) + direction
    new_node[2] = np.clip(new_node[2], 50, 300)
    return tuple(new_node)

def build_path(nodes, index):
    path = [nodes[index]]
    while index is not None:
        index = nodes[index][3]
        path.append(nodes[index])
    return path[::-1]

# Perform path planning
start = (15, 15, 25)
goal = tuple(destination)
drone_size = 30
nodes, path = rrt(start, goal, obstacles_kdtree, drone_size)

# Plot the nodes and path
for node in nodes:
    ax.plot([node[0], node[3][0]], [node[1], node[3][1]], [node[2], node[3][2]], 'b-', linewidth=0.5)

if path is not None:
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=2, label='Path')
else:
    print("Path not found")

plt.legend()
plt.show()