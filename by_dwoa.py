import numpy as np
import math
import matplotlib.pyplot as plt

class Double_MOWOA():
    def __init__(self, func1, func2,n_users, n_subcarriers, user_demands, available_spectrum, n_iter=50, n_pop=50, a=2, a_decay=0.9):
        self.func1 = func1  # 初始目标函数1
        self.func2 = func2  # 初始目标函数2
        self.n_users = n_users
        self.n_subcarriers = n_subcarriers
        self.user_demands = user_demands
        self.available_spectrum = available_spectrum
        self.n_iter = n_iter  # 最大迭代次数
        self.n_pop = n_pop  # 种群大小
        self.a = a  # a线性衰减系数
        self.a_decay = a_decay # a衰减率
        
        # 初始化最优解
        self.x1_best = 0 
        self.x2_best = 0 
        self.f1_best = float("inf")
        self.f2_best = float("inf")
        
    # 随机初始化种群 
    # def init_pop(self):
    #     pop = np.random.rand(self.n_pop, 2)  
    #     return pop
    # 
    def init_pop(self):
        x1 = np.random.rand(self.n_pop)
        x2 = np.random.rand(self.n_pop)
        pop = np.stack((x1, x2), axis=-1)
        return pop  

    # 目标函数,func1: eMBB吞吐量,func2: URLLC时延
    def eval_func(self, pop):
        x1 = pop[:, 0]  # eMBB频谱分配
        x2 = pop[:, 1]  # URLLC频谱分配
        
        f1 = x1  # eMBB吞吐量
        f2 = 1 - x2  # URLLC时延
        
        return f1, f2

    # 快速非劣解排序
    def fast_nondominated_sort(self, f1, f2): 
        n = len(f1)
        # 初始化每个解的支配个数和被支配解
        n_dom = np.zeros(n)  
        dominated = [[] for _ in range(n)]

        # 遍历每个解,判断支配关系
        for i in range(n):
            for j in range(n):  
                if (f1[i] > f1[j] and f2[i] > f2[j]) or (f1[i] >= f1[j] and f2[i] > f2[j]) or (f1[i] > f1[j] and f2[i] >= f2[j]):  
                    n_dom[j] += 1  # 被i支配,支配个数加1
                    dominated[i].append(j)  # i支配j

        # 获取不同非劣解层的解下标
        fronts = []  
        front = []
        Q = []  
        for i in range(n):  
            if n_dom[i] == 0: # 未被任何解支配,属于第一层非劣解
                front.append(i) 
                Q.append(i)  
        fronts.append(front)

        # 获得其余各层非劣解
        while Q:  
            # 新一层非劣解集         
            front = []  
            # 该层新加入的可供选择解的索引        
            Q_next = []  

            for i in Q:  
                for j in dominated[i]:  
                    n_dom[j] -= 1  # 被支配解的支配个数减1
                    if n_dom[j] == 0: # 转换为非劣解
                        Q_next.append(j)  
                        front.append(j)   # 添加至当前层非劣解集

            fronts.append(front)  
            Q = Q_next  # 重复上述步骤

        # Filter out empty fronts before returning them
        fronts = [front for front in fronts if front]

        return fronts  

    # 以非劣解获取下一代种群
    def gen_next_pop(self, pop, f1, f2, fronts):
        next_pop = []
        # 选择当前最佳非劣解
        best_f1 = float("inf"); best_idx = 0
        for i, front in enumerate(fronts):
            if f1[front[0]] <= best_f1:
                best_f1 = f1[front[0]]  
                best_idx = i  

        # 选择该层所有的非劣解
        chosen = fronts[best_idx]  
        for idx in chosen:  
            next_pop.append(pop[idx])    

        # 确保种群大小为n_pop,随机选择其余解
        while len(next_pop) < self.n_pop:  
            rnd = np.random.randint(0, len(pop))
            if rnd not in chosen:  
                next_pop.append(pop[rnd])  

        return np.array(next_pop) 
        
    def optimize(self):
        pop = self.init_pop()  # 初始化种群
        f1, f2 = self.eval_func(pop)  # 计算目标函数值
        
        # 搜索迭代
        for t in range(self.n_iter):  # 迭代次数
            fronts = self.fast_nondominated_sort(f1, f2) # 非劣解排序
            next_pop = self.gen_next_pop(pop, f1, f2, fronts) # 选择非劣解产生新种群
            
            # 线性衰减a值
            self.a = self.a * (1 - self.a_decay) 
            
            # 重新评估新种群的目标函数值
            f1, f2 = self.eval_func(next_pop)  
            
            # 更新最优解
            for i in range(len(f1)):
                if f1[i] <= self.f1_best and f2[i] <= self.f2_best:
                    self.f1_best = f1[i]  
                    self.f2_best = f2[i]  
                    self.x1_best = next_pop[i, 0]  
                    self.x2_best = next_pop[i, 1]
                    
            # 种群更新
            pop = next_pop 
            self.pop = pop 
            
        # 返回最优解  
        solution = [self.x1_best, self.x2_best]
        return solution
    
def func1(x1, x2):
    """
    n_users: 用户数量
    n_subcarriers: 子载波数量
    user_demands: 每个用户的带宽需求，单位为Mbps
    available_spectrum: 可用频谱资源，单位为MHz
    """
    n_users = int(x1)
    n_subcarriers = int(x2)

    # 初始化每个用户的子载波分配
    user_subcarrier_allocation = np.zeros((n_users, n_subcarriers))
    
    # 计算每个用户的子载波需求
    total_demand = sum(user_demands)
    subcarrier_demands = [n_subcarriers * (demand / total_demand) for demand in user_demands]
    
    # 分配子载波给每个用户
    allocated_subcarriers = 0
    for i in range(n_users):
        user_subcarrier_allocation[i, allocated_subcarriers:int(allocated_subcarriers + subcarrier_demands[i])] = 1
        allocated_subcarriers += int(subcarrier_demands[i])
    
    # 分配剩余的子载波
    remaining_subcarriers = n_subcarriers - allocated_subcarriers
    if remaining_subcarriers > 0:
        for i in range(remaining_subcarriers):
            user_subcarrier_allocation[i, allocated_subcarriers + i] = 1
    
    # 计算每个用户分配到的频谱资源
    user_spectrum_allocation = (available_spectrum / n_subcarriers) * np.sum(user_subcarrier_allocation, axis=1)
    
    return user_spectrum_allocation

def func2(n_users, n_subcarriers, user_demands, available_spectrum):
    """
    n_users: Number of users
    n_subcarriers: Number of subcarriers
    user_demands: A list of minimum bandwidth demands for each user, in Mbps
    available_spectrum: Available spectrum resources, in MHz
    """

    # Initialize each user's subcarrier allocation
    user_subcarrier_allocation = np.zeros((n_users, n_subcarriers))

    # Calculate each user's subcarrier demand
    total_demand = sum(user_demands)
    subcarrier_demands = [n_subcarriers * (demand / total_demand) for demand in user_demands]

    # Allocate subcarriers to each user to satisfy their minimum bandwidth requirements
    allocated_subcarriers = 0
    for i in range(n_users):
        user_subcarrier_allocation[i, allocated_subcarriers:int(allocated_subcarriers + subcarrier_demands[i])] = 1
        allocated_subcarriers += int(subcarrier_demands[i])

    # Allocate the remaining subcarriers to minimize latency
    remaining_subcarriers = n_subcarriers - allocated_subcarriers
    if remaining_subcarriers > 0:
        for i in range(remaining_subcarriers):
            user_subcarrier_allocation[i % n_users, allocated_subcarriers + i] = 1

    # Calculate each user's allocated spectrum resources
    user_spectrum_allocation = (available_spectrum / n_subcarriers) * np.sum(user_subcarrier_allocation, axis=1)

    # Calculate the latency for each user (assuming lower latency with higher allocated spectrum)
    user_latency = [1 / spectrum for spectrum in user_spectrum_allocation]

    # Calculate the average latency as the objective
    average_latency = sum(user_latency) / n_users

    return average_latency


# 实例化算法对象  
mooa = Double_MOWOA(func1, func2, n_iter=200, n_pop=100)  
# 多目标优化搜索  
solution = mooa.optimize()
# 最优解  
x1_best, x2_best = solution
print(x1_best, x2_best)  
# 绘图
x1 = []; x2 = []  
for i in range(mooa.n_pop):
    x1.append(mooa.pop[i, 0])
    x2.append(mooa.pop[i, 1])
plt.scatter(x1, x2)  
plt.plot(x1_best, x2_best, 'ro', markersize=10)    
plt.show()