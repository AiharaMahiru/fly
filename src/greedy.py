import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

random.seed(42)
np.random.seed(42)

# 创建三维场景
def create_box(x, y, z, dx, dy, dz):
    """Create a box with one corner at the given coordinates and with the given dimensions."""
    return [
        # Bottom
        [(x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z)],
        # Side 1
        [(x, y, z), (x, y, z + dz), (x + dx, y, z + dz), (x + dx, y, z)],
        # Side 2
        [(x, y, z), (x, y + dy, z), (x, y + dy, z + dz), (x, y, z + dz)],
        # Side 3
        [
            (x + dx, y, z),
            (x + dx, y + dy, z),
            (x + dx, y + dy, z + dz),
            (x + dx, y, z + dz),
        ],
        # Side 4
        [
            (x, y + dy, z),
            (x, y + dy, z + dz),
            (x + dx, y + dy, z + dz),
            (x + dx, y + dy, z),
        ],
        # Top
        [
            (x, y, z + dz),
            (x + dx, y, z + dz),
            (x + dx, y + dy, z + dz),
            (x, y + dy, z + dz),
        ],
    ]

# 三维盒子
class TreeNode:
    def __init__(self, point, parent=None):
        self.point = np.array(point)
        self.parent = parent

# 判断点是否在障碍物内
def point_in_obstacle(point, obstacles):
    for obstacle in obstacles:
        if (
            obstacle[0][0] <= point[0] <= (obstacle[0][0] + obstacle[1][0])
            and obstacle[0][1] <= point[1] <= (obstacle[0][1] + obstacle[1][1])
            and obstacle[0][2] <= point[2] <= (obstacle[0][2] + obstacle[1][2])
        ):
            return True
    return False

# 生成随机点
def generate_random_point(x_max, y_max, z_min, z_max):
    return np.array(
        [
            random.uniform(0, x_max),
            random.uniform(0, y_max),
            random.uniform(z_min, z_max),
        ]
    )

# 判断两点之间是否有障碍物
def is_obstructed(cell, obstacles, grid_size):
    for obstacle in obstacles:
        x_min, y_min, z_min = obstacle[0]
        dx, dy, dz = obstacle[1]
        x_max, y_max, z_max = x_min + dx, y_min + dy, z_min + dz

        if x_min <= cell[0] <= x_max and y_min <= cell[1] <= y_max:
            return True
    return False

# 创建地面网格
def create_ground_grid(x_max, y_max, grid_size, obstacles):
    grid = []
    for x in range(0, x_max, grid_size):
        for y in range(0, y_max, grid_size):
            cell = (x + grid_size // 2, y + grid_size // 2)
            if not is_obstructed(cell, obstacles, grid_size):
                grid.append(cell)
    return np.array(grid)

# Greedy算法实现
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
'''
函数 greedy_traversal(开始点,目标点,地面网格):

初始化未访问集合为ground_grid中的所有索引

初始化路径列表为开始点

初始化当前索引current_index为None(初始最近点)

找到最近的开始点:

对ground_grid中的每个格子:
*计算与开始点的距离
*如果更近则更新最小距离min_distance和当前索引current_index
将current_index从未访问集合中移除

初始化start_z和goal_z分别从开始点和目标点获得

根据开始点到目标点的相对距离计算下一个点的z值

将下一个点添加到路径列表中

若未访问集合不为空:

初始化最小距离min_distance 为无穷大

初始化下一个索引next_index 为None(下一个最近点)

对未访问集合中的每个索引i:

计算与路径列表中前一个点的距离
如果更近则更新最小距离min_distance 和下一个索引next_index
将next_index从未访问集合中移除

将当前索引current_index赋值为next_index

根据开始点到目标点的相对距离计算下一个点的z值

将下一个点添加到路径列表中
'''


# 梯度下降法顺滑路径
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
'''
函数gradient_descent_path_smooth(路径、障碍物信息、α、β、最大迭代次数、公差):

把路径转换成浮点数

定义path_cost()函数计算基于平滑度和障碍物的成本

定义gradient()函数根据平滑度和障碍物在每个点计算梯度

把路径复制给optimized_path

对最大迭代次数:

把optimized_path复制给new_path
对new_path中的每一个点i(除端点):
计算该点的梯度
通过减去α乘以梯度来更新new_path[i]
如果 new_path 和 optimized_path之间的距离小于公差:
break
赋值optimized_path = new_path
返回optimized_path
'''

# 生成障碍物
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

# 在场景中添加障碍物
def create_obstacles(ax, num_obstacles, x_lim, y_lim, min_distance, start, goal):
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

    for obstacle in obstacles_info:
        box = Poly3DCollection(
        create_box(*obstacle[0], *obstacle[1]), alpha=0.3, linewidths=1, edgecolors="k"
    )
        box.set_facecolor("gray")
        ax.add_collection3d(box)
    return obstacles_info

# 显示环境
def show_env(start, goal, obstacles_info, path):
    path = np.array(path + [goal])
    start_height = start[2]
    goal_height = goal[2]
    heights = np.linspace(start_height, goal_height, len(path))
    path[:, 2] = heights  # Set the height to the spraying height

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
    ax1.plot(smoothed_path[:, 0], smoothed_path[:, 1], smoothed_path[:, 2], "r-", linewidth=3, label="Optimized 3D Path")
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

def main():
    # 初始化环境
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # 定义函数参数
    num_obstacles = 15   # 障碍物数量
    x_lim, y_lim = 950, 950  # 场景大小
    min_distance = 200  # 障碍物最小间距

    start = np.array([50, 50, 100])  # 起点
    goal = np.array([900, 900, 300])  # 终点

    # 障碍物坐标列表
    obstacles_info = create_obstacles(ax, num_obstacles, x_lim, y_lim, min_distance, start, goal)

    # 场景网格化
    ground_grid = create_ground_grid(1000, 1000, 80, obstacles_info)

    # 路径规划
    path = greedy_traversal(start, goal, ground_grid)

    # 显示环境
    show_env(start, goal, obstacles_info, path)

# debug
# print("obstacles_info: ", type(obstacles_info))

if __name__ == "__main__":
    main()
    
'''
流程：

初始化环境 - 创建3D图形和轴,定义参数如障碍物数量、场景大小等

生成障碍物 - 使用给定的限制生成随机障碍物

创建地面网格 - 在不包含障碍物的区域创建网格

找出路径 - 使用贪心遍历算法找到起点到目标点的初始路径

平滑路径 - 使用梯度下降方法平滑路径以避开障碍物

显示环境 - 在3D和2D图中显示 paths 和障碍物

主函数 - 定义参数并按照上述步骤执行:

生成障碍物
创建地面网格
找到初始路径
平滑最优路径
显示结果环境
调用主函数运行整个过程,生成优化后的3D和2D路径
'''