import numpy as np
import matplotlib.pyplot as plt
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
plt.show()