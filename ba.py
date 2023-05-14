import numpy as np
from fen import fen
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)  # 设置随机种子以保证可重复性

# 定义目标函数
def objective_function(individual):
    x1, x2 = individual
    avg_sensor_delay, avg_video_speed = fen(x1, x2)

    if avg_sensor_delay > 3 or avg_video_speed < 250:
        return float("inf"), 0

    return -avg_sensor_delay, avg_video_speed

# 创建优化问题的类型
creator.create(
    "FitnessMax", base.Fitness, weights=(-1.0, 1.0)
)  # 我们希望最小化avg_sensor_delay并最大化avg_video_speed
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# 设置遗传算法
toolbox = base.Toolbox()

# 注册属性生成器（x1和x2）的范围
x1_min, x1_max = 1, 100
x2_min, x2_max = 1, 100
toolbox.register("x1", np.random.randint, x1_min, x1_max + 1)
toolbox.register("x2", np.random.randint, x2_min, x2_max + 1)

# 注册个体生成器
toolbox.register(
    "individual", tools.initCycle, creator.Individual, (toolbox.x1, toolbox.x2), n=1
)

# 注册种群生成器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册评价函数
toolbox.register("evaluate", objective_function)

# 注册遗传操作符
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register(
    "mutate",
    tools.mutUniformInt,
    low=min(x1_min, x2_min),
    up=max(x1_max, x2_max),
    indpb=1,
)
toolbox.register("select", tools.selNSGA2)

# 设置主函数
def main():
    # 遗传算法的参数
    population_size = 100
    crossover_probability = 0.7
    mutation_probability = 0.2
    number_of_generations = 500

    # 创建初始种群
    population = toolbox.population(n=population_size)

    # 运行遗传算法
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_probability,
        mutpb=mutation_probability,
        ngen=number_of_generations,
        verbose=False,
    )

    # 提取最优个体
    best_individual = tools.selBest(population, 1)[0]
    best_x1, best_x2 = best_individual
    best_avg_sensor_delay, best_avg_video_speed = objective_function(best_individual)

    print("Best x1: {:.0f}, Best x2: {:.0f}".format(best_x1, best_x2))
    print("Best Average Sensor Delay: {:.2f} ms".format(-best_avg_sensor_delay))
    print("Best Average Video Speed: {:.2f} Mbps".format(best_avg_video_speed))

if __name__ == "__main__":
    main()


'''
目的：可优化的变量是 x1 和 x2,目标是最小化传感器延迟和最大化视频速率。
流程：

设置随机种子以保证重复性

定义目标函数:

调用fen()函数,传入x1和x2参数
计算平均传感器延迟和平均视频速率
如果延迟超过3ms或速率低于250Mbps,返回无穷大
否则返回:负延迟作为第一个权重,-速率作为第二个权重
创建个体类型:
FitnessMax:目标是最小化延迟和最大化速率
Individual:为np.ndarray, fitness为FitnessMax
设置遗传算法工具:
注册x1和x2的范围
注册个体生成器
注册种群生成器
注册目标函数
注册遗传算子:交叉/变异/选择
主函数:
设置参数:种群大小、交叉概率、变异概率、世代数
创建初始种群
运行遗传算法,记录日志
提取最优个体:最小延迟和最大速率
打印个体参数及结果
调用主函数运行整个过程

'''