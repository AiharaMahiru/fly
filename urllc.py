import numpy as np
import random

def uRLLC_spectrum_allocation(n_users, n_subcarriers, user_demands, user_latencies, available_spectrum):
    """
    n_users: 用户数量
    n_subcarriers: 子载波数量
    user_demands: 每个用户的带宽需求，单位为Mbps
    user_latencies: 每个用户的延迟要求，单位为ms
    available_spectrum: 可用频谱资源，单位为MHz
    """
    
    # 初始化每个用户的子载波分配
    user_subcarrier_allocation = np.zeros((n_users, n_subcarriers))

    # 计算每个用户的子载波需求，考虑带宽需求和延迟要求
    demands_latencies = [demand / latency for demand, latency in zip(user_demands, user_latencies)]
    total_demand_latency = sum(demands_latencies)
    subcarrier_demands = [n_subcarriers * (demand_latency / total_demand_latency) for demand_latency in demands_latencies]

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
    # Calculate the latency for each user (assuming lower latency with higher allocated spectrum)
    user_latency = [1 / spectrum for spectrum in user_spectrum_allocation]

    # Calculate the average latency as the objective
    average_latency = sum(user_latency) / n_users

    return user_spectrum_allocation, average_latency

if __name__ == "__main__":
    n_users = 300
    n_subcarriers = 3000
    user_demands = [random.randint(0, 5) * 100 + 50 for _ in range(300)]  # 用户带宽需求
    user_latencies = [random.randint(1, 5)*3  + 5 for _ in range(300)]  # 用户延迟要求
    available_spectrum = 100  # 可用频谱资源

    user_spectrum_allocation, average_latency = uRLLC_spectrum_allocation(n_users, n_subcarriers, user_demands, user_latencies, available_spectrum)
    # print("Spectrum allocation for each user (MHz):", user_spectrum_allocation)
    print("平均延时 (ms):", average_latency)