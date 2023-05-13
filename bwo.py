import numpy as np
import math
import matplotlib.pyplot as plt

class Double_MOWOA():
    def __init__(self, func1, func2, n_iter=50, n_pop=50, a=2, a_decay=0.9):
        self.func1 = func1  # 初始目标函数1
        self.func2 = func2  # 初始目标函数2
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
    def init_pop(self):
        pop = np.random.rand(self.n_pop, 2)  
        return pop  

    # 目标函数评估
    def eval_func(self, pop):
        f1 = [self.func1(x1, x2) for x1, x2 in pop]
        f2 = [self.func2(x1, x2) for x1, x2 in pop]
        return np.array(f1), np.array(f2)
    
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
            
        # 返回最优解  
        solution = [self.x1_best, self.x2_best]
        return solution
    
def func1(x1, x2): 
    return x1  # 最大化x1

def func2(x1, x2):
    return 1-x2  # 最大化1-x2

def embb(x1, x2, x3, x4): #吞吐量 = 带宽 × 调制编码效率 × 资源利用率 x 用户量
    '''
    x1 = ue
    x2 = bandwidth
    x3 = QAM
    x4 = resource_u
    '''
    ue = x1
    bandwidth = x2
    QAM = x3
    resource_u = x4
    return ue * bandwidth * QAM * resource_u

def calc_emb_throughput(alloc_scheme):
    # alloc_scheme为eMBB和URLLC分配的频谱带宽,单位为Hz
    BW_eMBB = alloc_scheme[0]  
    BW_URLLC = alloc_scheme[1]
    
    # eMBB小区参数
    eMBB_cell_radius = 1000     # 小区半径,m 
    eMBB_snr = 10               # 信噪比,dB
    
    # 系统参数
    N0 = -174                  # 热噪声功率谱密度,dBm/Hz
    B = 40e6                      # 系统带宽,Hz
    
    # 计算小区内用户平均吞吐量 
    emb_cell_tp = BW_eMBB * math.log2(1 + (eMBB_snr * N0) / B)  
    
    # eMBB小区内用户数
    emb_cell_users = (3.14 * eMBB_cell_radius**2) / (4 * 3.14)  
    
    # eMBB系统吞吐量
    system_emb_tp = emb_cell_tp * emb_cell_users  
    
    return system_emb_tp   # 返回eMBB系统吞吐量,bit/s

#总时延 = 传输时延 + 处理时延 + 排队时延 + 其他时延
def urllc(x1, x2, x3, x4): #可靠性 = 1 - 误块率
    ue = x1
    bandwidth = x2
    QAM = x3
    resource_u = x4
    return 1 - ue * bandwidth * QAM * resource_u

def calc_url_latency(alloc_scheme):
    # alloc_scheme为eMBB和URLLC分配的频谱带宽,单位为Hz
    BW_eMBB = alloc_scheme[0]  
    BW_URLLC = alloc_scheme[1]
    
    # URLLC小区参数 
    url_cell_radius = 100        # 小区半径,m
    url_required_snr = 5        # URLLC用户需求SNR,dB
    
    # 系统参数(同上)
    N0 = -174                 
    B = 40e6                      
    
    # 计算URLLC用户最大允许路径损耗 
    max_path_loss = url_required_snr * N0 - 10*math.log10(BW_URLLC)  
    
    # 计算URLLC小区边缘用户与基站的距离
    edge_user_dist = (max_path_loss / (36.7 * 10**-3)) ** 0.5  
    
    # 考虑时延成分:传播时延 + 发送时延 + 接收时延 
    url_latency = (edge_user_dist / 3e8) + (1 / BW_URLLC) + 1e-6  

    return url_latency  # 返回URLLC时延,s

mooa = Double_MOWOA(func1, func2, n_iter=100, n_pop=50)

solution = mooa.optimize()

x1_best, x2_best = solution 
print(x1_best, x2_best)

# 绘制搜索空间Scatter diagram
x1 = []; x2 = [] 
for i in range(mooa.n_pop):
    x1.append(mooa.pop[i, 0])
    x2.append(mooa.pop[i, 1])
plt.scatter(x1, x2) 
plt.plot(x1_best, x2_best, 'ro', markersize=10) 
plt.show()