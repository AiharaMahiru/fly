import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.gca(projection='3d')

# 设置坐标轴标签和范围
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_zlim(0, 2 * spraying_height)

# 绘制障碍物
for obstacle in grid_map.obstacles:
    plot_obstacle(ax, obstacle)

drone_path, = ax.plot([], [], [], linewidth=2, color='blue', marker='o', markersize=5)

path_with_height = [(x, y, spraying_height) for (x, y) in path]
drone_positions = []

def update(i):
    global drone_positions
    start = path_with_height[i]
    goal = path_with_height[i + 1]
    came_from, cost_so_far = a_star_search(grid_map, start, goal)
    path_segment = reconstruct_path(came_from, start, goal)
    drone_positions.extend(path_segment[:-1])

    xs = [node[0] for node in drone_positions]
    ys = [node[1] for node in drone_positions]
    zs = [node[2] for node in drone_positions]
    drone_path.set_data_3d(xs, ys, zs)

    return drone_path,

anim = FuncAnimation(fig, update, frames=len(path_with_height) - 1, interval=200, blit=True)

plt.show()