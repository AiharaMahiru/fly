import random
import math
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font",family='YouYuan')
from fen import fen

class Double_MOWOA():
    def __init__(self, func, n_iter=50, n_pop=50, a=2, a_decay=0.9):
        self.func = func  # 目标函数
        self.n_iter = n_iter  # 最大迭代次数
        self.n_pop = n_pop  # 种群大小
        self.a = a  # a线性衰减系数
        self.a_decay = a_decay  # a衰减率

        # 初始化最优解
        self.x_best = None
        self.y_best = None
        
    # 随机初始化种群 
    def init_pop(self):
        pop = np.random.randint(0, 100, size=(self.n_pop, 2))  # Modify the range and size as needed
        return pop  

    # 目标函数评估
    def eval_func(self, pop):
        avg_sensor_delay = []
        avg_video_speed = []
        for x1, x2 in pop:
            sensor_delay, video_speed = self.func(int(x1), int(x2))
            avg_sensor_delay.append(sensor_delay)
            avg_video_speed.append(video_speed)
        return np.array(avg_sensor_delay), np.array(avg_video_speed)
    
    # 快速非劣解排序
    def fast_nondominated_sort(self, y1, y2):
        # Implementation omitted for brevity
        pass

    # 以非劣解获取下一代种群
    def gen_next_pop(self, pop, y1, y2, fronts):
        # Implementation omitted for brevity
        pass
        
    def optimize(self):
        pop = self.init_pop()  # 初始化种群
        
        if pop is None:
            raise ValueError("Population is None.")
        
        y1, y2 = self.eval_func(pop)  # 计算目标函数值
        
        # 搜索迭代
        for t in range(self.n_iter):  # 迭代次数
            fronts = self.fast_nondominated_sort(y1, y2) # 非劣解排序
            next_pop = self.gen_next_pop(pop, y1, y2, fronts) # 选择非劣解产生新种群
            
            # 线性衰减a值
            self.a = self.a * (1 - self.a_decay) 
            
            # 重新评估新种群的目标函数值
            y1, y2 = self.eval_func(next_pop)  
            
            # 更新最优解
            best_idx = np.argmin(y1)  # Find the index of the best solution based on y1 (minimize y1)
            self.x_best = next_pop[best_idx]  # Store the best solution
            self.y_best = (y1[best_idx], y2[best_idx])  # Store the best objective values
            
            # 种群更新
            pop = next_pop  
            
        # 返回最优解  
        return self.x_best, self.y_best

mooa = Double_MOWOA(fen, n_iter=100, n_pop=50)

solution_x, solution_y = mooa.optimize()

x1_best, x2_best = solution_x
avg_sensor_delay_best, avg_video_speed_best = solution_y
print('x1_best:', x1_best)
print('x2_best:', x2_best)
print('avg_sensor_delay_best:', avg_sensor_delay_best)
print('avg_video_speed_best:', avg_video_speed_best)
