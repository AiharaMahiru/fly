import numpy as np
import matplotlib.pyplot as plt

def generate_spiral_path(x_range, y_range, step_x, step_y):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x, y = x_min, y_min
    path = [(x, y)]

    while x_max > x_min and y_max > y_min:
        # Move in the positive X direction
        for x in np.arange(x + step_x, x_max, step_x):
            path.append((x, y))

        x_max -= step_x

        # Move in the positive Y direction
        for y in np.arange(y + step_y, y_max, step_y):
            path.append((x, y))

        y_max -= step_y

        # Move in the negative X direction
        for x in np.arange(x - step_x, x_min - step_x, -step_x):
            path.append((x, y))

        x_min += step_x

        # Move in the negative Y direction
        for y in np.arange(y - step_y, y_min - step_y, -step_y):
            path.append((x, y))

        y_min += step_y

    return path

# Define the field dimensions and step sizes
x_range = (0, 100)
y_range = (0, 100)
step_x = 5
step_y = 5

# Generate the spiral path
path = generate_spiral_path(x_range, y_range, step_x, step_y)

# Set the spraying height
spraying_height = 5

# Add the spraying height (Z value) to the path
path_with_height = [(x, y, spraying_height) for (x, y) in path]

# Now you can use this path to guide the drone for spraying


# 使用之前的 generate_spiral_path 函数生成螺旋路径
path = generate_spiral_path(x_range, y_range, step_x, step_y)

# 将路径的X和Y坐标分别存储在列表中
path_x = [point[0] for point in path]
path_y = [point[1] for point in path]

# 创建一个matplotlib图形，显示螺旋路径
fig, ax = plt.subplots()
ax.plot(path_x, path_y, marker='o', markersize=3, linestyle='-')
ax.set_title("Drone Spraying Path")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

# 设置坐标轴范围
ax.set_xlim(x_range)
ax.set_ylim(y_range)

# 显示图形
plt.show()