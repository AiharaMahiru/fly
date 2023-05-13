import numpy as np
import random

def eMBB_spectrum_allocation(n_users, n_subcarriers, user_demands, available_spectrum):
    """
    n_users: 用户数量
    n_subcarriers: 子载波数量
    user_demands: 每个用户的带宽需求，单位为Mbps
    available_spectrum: 可用频谱资源，单位为MHz
    """
    
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

def uRLLC_spectrum_allocation(n_users, n_subcarriers, user_demands, available_spectrum):
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
    user_latency = []
    for spectrum in user_spectrum_allocation:
        if spectrum == 0:
            user_latency.append(float("inf"))
        else:
            user_latency.append(1 / spectrum)

    # Calculate the average latency as the objective
    average_latency = sum(user_latency) / n_users

    return average_latency, user_spectrum_allocation

def s_uRLLC_spectrum_allocation(n_users, n_subcarriers, user_demands, available_spectrum):
    # Initialize each user's subcarrier allocation
    user_subcarrier_allocation = np.zeros((n_users, n_subcarriers))

    # Calculate each user's subcarrier demand
    total_demand = sum(user_demands)
    subcarrier_demands = [n_subcarriers * (demand / total_demand) for demand in user_demands]

    # Sort users by the inverse of their demands
    sorted_users = sorted(range(n_users), key=lambda i: 1 / user_demands[i])

    # Allocate subcarriers to each user to satisfy their minimum bandwidth requirements
    allocated_subcarriers = 0
    for i in sorted_users:
        user_subcarrier_allocation[i, allocated_subcarriers:int(allocated_subcarriers + subcarrier_demands[i])] = 1
        allocated_subcarriers += int(subcarrier_demands[i])

    # Calculate each user's allocated spectrum resources
    user_spectrum_allocation = (available_spectrum / n_subcarriers) * np.sum(user_subcarrier_allocation, axis=1)

    # Calculate the latency for each user (assuming lower latency with higher allocated spectrum)
    user_latency = []
    for spectrum in user_spectrum_allocation:
        if spectrum == 0:
            user_latency.append(float("inf"))
        else:
            user_latency.append(1 / spectrum)

    # Calculate the average latency as the objective
    average_latency = sum(user_latency) / n_users

    return average_latency, user_spectrum_allocation

if __name__ == "__main__":
    '''
    总带宽 (Hz) / 子载波间隔 (Hz) = 子载波数量
    '''
    n_users = 1000
    n_subcarriers = 7000
    user_demands = [random.randint(0, 5) * 100 + 50 for _ in range(1000)]
    available_spectrum = 100

    eMBB_allocation = eMBB_spectrum_allocation(n_users, n_subcarriers, user_demands, available_spectrum)
    uRLLC_average_latency, user_spectrum_allocation= s_uRLLC_spectrum_allocation(n_users, n_subcarriers, user_demands, available_spectrum)
    print("eMBB spectrum allocation: ", eMBB_allocation)
    print("uRLLC average latency: ", uRLLC_average_latency)
    print("uRLLC spectrum allocation: ", user_spectrum_allocation)