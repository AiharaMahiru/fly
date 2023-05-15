import numpy as np
from resource_allocation import fen, plot_metrics
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)  # 设置随机种子以保证可重复性

# 定义目标函数
def objective_function(individual):
    x1, x2 = individual
    avg_sensor_delay,  avg_video_speed, sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation = fen(x1, x2)

    if avg_sensor_delay > 3 or avg_video_speed < 250:
        return float("inf"), 0

    individual.attributes = (sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation)
    return -avg_sensor_delay, avg_video_speed

# 创建优化问题的类型
creator.create(
    "FitnessMax", base.Fitness, weights=(-1.0, 1.0)
)  # 我们希望最小化avg_sensor_delay并最大化avg_video_speed
# creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, attributes=None)

# 设置遗传算法
toolbox = base.Toolbox()

# 注册属性生成器（x1和x2）的范围
x1_min, x1_max = 1, 1000
x2_min, x2_max = 1, 1000
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

def main(population_size, crossover_probability, mutation_probability, number_of_generations):

    print_frequency = 50

    # 创建初始种群
    population = toolbox.population(n=population_size)

    # 准备存储每100代的结果
    all_avg_sensor_delays = []
    all_avg_video_speeds = []

    # 运行遗传算法
    for gen in range(1, number_of_generations + 1):
        population = algorithms.varAnd(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability)
        fits = toolbox.map(toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit

        population = toolbox.select(population, k=len(population))


        if gen % print_frequency == 0:
            best_individual = tools.selBest(population, 1)[0]
            best_avg_sensor_delay, best_avg_video_speed = objective_function(best_individual)
            sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation = best_individual.attributes
            if best_avg_sensor_delay != float('inf') and best_avg_video_speed != 0:
                all_avg_sensor_delays.append(-best_avg_sensor_delay)
                all_avg_video_speeds.append(best_avg_video_speed)
                print(f"Generation {gen} | Best Average Sensor Delay: {-best_avg_sensor_delay:.2f} ms | Best Average Video Speed: {best_avg_video_speed:.2f} Mbps")


    # 找到最优解所在的代数以及对应的 avg_sensor_delay 和 avg_video_speed
    best_generation = (all_avg_sensor_delays.index(min(all_avg_sensor_delays)) + 1) * print_frequency
    best_sensor_delay = min(all_avg_sensor_delays)
    best_video_speed = all_avg_video_speeds[all_avg_sensor_delays.index(min(all_avg_sensor_delays))]

    best_individual_x1, best_individual_x2 = best_individual

    # 绘制结果图像
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(range(print_frequency, len(all_avg_sensor_delays) * print_frequency + 1, print_frequency), all_avg_sensor_delays, label="Best Average Sensor Delay")
    ax1.scatter(best_generation, best_sensor_delay, color="red", marker="o", label="Best Solution")
    ax1.annotate(f"x1={best_individual_x1:.2f}, x2={best_individual_x2:.2f},\nBest Avg. Sensor Delay={best_avg_sensor_delay:.2f}",
                xy=(best_generation, best_sensor_delay),
                xytext=(best_generation - 0.4 * best_generation, best_sensor_delay + 0.1 * best_sensor_delay),
                arrowprops=dict(facecolor="black", shrink=0.05),
                fontsize=9)
    ax1.set_ylabel("Sensor Delay (ms)")
    ax1.legend()

    ax2.plot(range(print_frequency, len(all_avg_video_speeds) * print_frequency + 1, print_frequency), all_avg_video_speeds, label="Best Average Video Speed", color="orange")
    ax2.scatter(best_generation, best_video_speed, color="red", marker="o", label="Best Solution")
    ax2.annotate(f"x1={best_individual_x1:.2f}, x2={best_individual_x2:.2f},\nBest Avg. Video Speed={best_avg_video_speed:.2f}",
                xy=(best_generation, best_video_speed),
                xytext=(best_generation - 0.4 * best_generation, best_video_speed - 0.1 * best_video_speed),
                textcoords="data",
                arrowprops=dict(facecolor="black", shrink=0.05),
                fontsize=9)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Video Speed (Mbps)")
    ax2.legend()
    
    plt.show()

    # plot_metrics(sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation)

if __name__ == "__main__":
    '''
    population_size : 种群大小
    crossover_probability : 交叉概率
    mutation_probability : 变异概率
    number_of_generations : 迭代次数
    '''
    # 遗传算法的参数
    population_size = 100
    crossover_probability = 0.7
    mutation_probability = 0.2
    number_of_generations = 5000

    

    main(population_size, crossover_probability, mutation_probability, number_of_generations)
    

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