import random
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rc("font",family='YouYuan')

random.seed(42)
np.random.seed(42)

class DummyDrone:
    def __init__(self, index):
        self.index = index
        self.slices = {}

    def connect(self, network_slice, allocation):
        self.slices[network_slice.slice_type] = allocation

class DummyNetworkSlice:
    def __init__(self, slice_type, resources):
        self.slice_type = slice_type
        self.resources = resources

'''
类 DummyDrone (为硬件调试预留接口)

初始化 DummyDrone 类的实例，设置索引
定义连接函数：将网络切片和分配连接到无人机

类 DummyNetworkSlice

初始化 DummyNetworkSlice 类的实例，设置切片类型和资源
'''

def create_network_slice(slice_type, resources):
    return DummyNetworkSlice(slice_type, resources)

'创建实例：网络切片'

def calculate_bandwidth(subcarrier_count, subcarrier_spacing):
    bandwidth = subcarrier_count * subcarrier_spacing
    return bandwidth/1e6

'计算带宽：子载波数量 * 子载波间距     返回带宽（以 MHz 为单位）'

def bandwidth_allocation(n_services, service_demands, available_bandwidth):
    demand_matrix = np.array(service_demands).reshape((2, -1)).T

    # 构建线性规划模型
    c = np.zeros(n_services)
    bounds = [(0, None)] * n_services
    constraints = [{'type': 'eq', 'fun': lambda x, i=i: np.sum(x[i::n_services]) - demand_matrix[i // 2, i % 2]}
                   for i in range(n_services)]

    # 求解线性规划问题
    res = linprog(c, bounds=bounds, A_eq=np.eye(n_services), b_eq=demand_matrix.flatten(), method='highs')

    resource_allocation = res.x

    return resource_allocation

'''
将服务需求重塑为需求矩阵
构建线性规划模型
求解线性规划问题
返回资源分配结果
'''

def calculate_video_delay(video_w, subcarrier_spacing, v_packet_size, distance, mimo):
    # sensor_bandwidth = calculate_bandwidth(sensor_subcarriers, subcarrier_spacing)
    video_bandwidth = video_w

    # video_speed = 1 / video_bandwidth
    video_speed = video_bandwidth * subcarrier_spacing * 1e3
    # print(video_speed)

    # sensor_transmission_time = sensor_symbol_duration * packet_size
    video_transmission_time = v_packet_size / (video_speed * mimo)

    propagation_delay = distance / 3e8 * 1e3  # 假设无人机之间的传播速度为光速（3e8 m/s）

    random_delay = random.choice([5e-3, 10e-3, 15e-3])
    
    video_delay = video_transmission_time + propagation_delay + random_delay
    # print(video_delay)

    return video_delay*1e3

'''
计算视频带宽
计算视频传输速度
计算视频传输时间
计算传播延迟
随机选择延迟(模拟抖动)
计算视频延迟
返回视频延迟（以毫秒为单位）
'''

def calculate_sensor_delay(sensor_w, subcarrier_spacing, s_packet_size, distance, mimo):
    sensor_bandwidth = sensor_w

    sensor_speed = sensor_bandwidth * subcarrier_spacing * 1e3
    # print(sensor_speed)

    # sensor_transmission_time = sensor_symbol_duration * packet_size
    sensor_transmission_time = s_packet_size / (sensor_speed * mimo)

    propagation_delay = distance / 3e8 * 1e3  # 假设无人机之间的传播速度为光速（3e8 m/s）

    random_delay = random.choice([1e-3, 2e-3, 3e-3])
    
    sensor_delay = sensor_transmission_time + propagation_delay + random_delay
    # print(sensor_delay)

    return sensor_delay*1e3

'''
计算传感器带宽
计算传感器传输速度
计算传感器传输时间
计算传播延迟
随机选择延迟(模拟抖动)
计算传感器延迟
返回传感器延迟（以毫秒为单位）
'''

def calculate_speed(bandiwidth, subcarrier_spacing, mimo):
    speed = bandiwidth * subcarrier_spacing * mimo
    return speed / 1e3
'计算速度：带宽 * 子载波间距 * MIMO   返回速度（以 Mbps 为单位）'

def plot_metrics(sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation):
    n_drones = len(sensor_bandwidth)
    # print(n_drones)
    drone_indices = list(range(n_drones))
    # print(drone_indices)

    x_ticks = np.arange(1, n_drones+1)
    x_positions = np.arange(n_drones)

    fig, axs = plt.subplots(2, 2, sharex=True)

    ax0, ax1 = axs[0]
    ax2, ax3 = axs[1]

    ax0.bar(drone_indices, sensor_allocation, width=0.4, label='Sensor')
    ax0.bar([i + 0.4 for i in drone_indices], video_allocation, width=0.4, label='Video')

    fig.suptitle('无人机传感器和视频信号的资源分配')
    fig.supxlabel('无人机编号')
    ax0.set_xticks(x_positions)
    ax0.set_xticklabels(x_ticks)
    ax0.set_ylabel('资源分配(子载波数量)')

    ax1.bar(drone_indices, sensor_bandwidth, width=0.4, label='Sensor')
    ax1.bar(drone_indices, video_bandwidth, width=0.4, bottom=sensor_bandwidth, label='Video')
    ax1.set_ylabel('分配频率带宽 (MHz)')

    ax2.bar(drone_indices, sensor_delay, width=0.4, label='Sensor')
    ax2.bar(drone_indices, video_delay, width=0.4, bottom=sensor_delay, label='Video')
    # ax2.set_xlabel('无人机')
    ax2.set_ylabel('预计单向延迟 (ms)')

    ax3.bar(drone_indices, sensor_speed, width=0.4, label='Sensor')
    ax3.bar(drone_indices, video_speed, width=0.4, bottom=sensor_delay, label='Video')
    # ax3.set_xlabel('无人机')
    ax3.set_ylabel('预计速率 (Mbps)')

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()

'''
设置图表参数, 两行两列
绘制每个指标的柱状图
显示图表
'''

def qos_based_allocation(n_services, service_demands, available_resources, performance_requirements):
    resource_allocation = [0] * n_services

    # 随机分配资源作为示例
    for i in range(n_services):
        resource_allocation[i] = random.uniform(service_demands[i] * 0.5, service_demands[i] * 1.5)

    # 根据可用资源按比例调整分配
    total_allocated = sum(resource_allocation)
    scaling_factor = available_resources / total_allocated
    resource_allocation = [x * scaling_factor for x in resource_allocation]

    # 根据性能要求进行调整
    for i in range(n_services):
        if i < len(performance_requirements):
            if 'max_latency' in performance_requirements[i]:
                max_latency = performance_requirements[i]['max_latency']
                # 进行相应的资源调整

            if 'min_bandwidth' in performance_requirements[i]:
                min_bandwidth = performance_requirements[i]['min_bandwidth']
                # 进行相应的资源调整

            if 'packet_loss_rate' in performance_requirements[i]:
                packet_loss_rate = performance_requirements[i]['packet_loss_rate']
                # 进行相应的资源调整

    return resource_allocation

'''
初始化资源分配
随机分配资源作为示例
根据可用资源按比例调整分配
根据性能要求进行调整
返回资源分配结果
'''

def fen(x1, x2):
    '''
    x1:传感器抽象
    x2:视频抽象
    '''
    #无人机数量
    n_drones = 20
    drones = [DummyDrone(i) for i in range(n_drones)]

    #划分传输比例
    sensor_demands = [x1] * n_drones
    video_demands = [x2] * n_drones

    bandwidth = 100e6  # 频宽为100MHz
    subcarrier_spacing = 15e3  # 子载波间隔为15kHz

    # 计算总的子载波数量
    total_subcarriers = int(bandwidth / subcarrier_spacing)

    available_resources = total_subcarriers

    performance_requirements = [
        {'max_latency': 5e-3, 'packet_loss_rate': 0.001},
        {'min_bandwidth': 50e6, 'packet_loss_rate': 0.01}
    ]

    resource_allocation = qos_based_allocation(len(sensor_demands) + len(video_demands), sensor_demands + video_demands, available_resources, performance_requirements)

    # print('可用资源：', available_resources)

    sensor_allocation = resource_allocation[:n_drones]
    video_allocation = resource_allocation[n_drones:]

    slice_1 = create_network_slice(slice_type='sensor', resources=sensor_allocation)
    slice_2 = create_network_slice(slice_type='video', resources=video_allocation)

    for i in range(n_drones):
        drone = drones[i]
        drone.connect(slice_1, allocation=sensor_allocation[i])
        drone.connect(slice_2, allocation=video_allocation[i])

    # 计算传感器和视频信号的带宽和延迟
    sensor_bandwidth = [calculate_bandwidth(subcarrier_count, subcarrier_spacing) for subcarrier_count in sensor_allocation]
    video_bandwidth = [calculate_bandwidth(subcarrier_count, subcarrier_spacing) for subcarrier_count in video_allocation]

    # 不同服务的包大小
    v_packet_size = 50e6
    s_packet_size = 15e3

    # mimo数
    mimo = 4

    # 预设延时和传输速度
    sensor_delay = []
    video_delay = []
    video_speed = []
    sensor_speed = []

    # 假设无人机之间的距离为100米
    drone_distance = 100

    for i in range(n_drones):
        # sensor_subcarriers = sensor_allocation[i]
        # video_subcarriers = video_allocation[i]

        sensor_w = sensor_bandwidth[i]
        video_w = video_bandwidth[i]

        sensor_delay.append(calculate_sensor_delay(sensor_w, subcarrier_spacing, s_packet_size, drone_distance,mimo))
        video_delay.append(calculate_video_delay(video_w, subcarrier_spacing, v_packet_size, drone_distance, mimo))

        video_speed.append(calculate_speed(video_w, subcarrier_spacing, mimo))
        sensor_speed.append(calculate_speed(sensor_w, subcarrier_spacing, mimo))

        # print('无人机{}的传感器信号延迟为{}ms，视频信号延迟为{}ms'.format(i, sensor_delay[i], video_delay[i]))
        # print('无人机{}的传感器信号速度为{}mbps，视频信号速度为{}mbps'.format(i, sensor_speed[i], video_speed[i]))

    # 计算平均延迟和平均传输速度
    avg_sensor_delay = np.mean(sensor_delay)
    avg_video_delay = np.mean(video_delay)
    avg_video_speed = np.mean(video_speed)
    avg_sensor_speed = np.mean(sensor_speed)

    # print('Average Sensor Delay: {:.2f} ms'.format(avg_sensor_delay))
    # print('Average Sensor Speed: {:.2f} Mbps'.format(avg_sensor_speed))
    # print('Average Video Delay: {:.2f} ms'.format(avg_video_delay))
    # print('Average Video Speed: {:.2f} Mbps'.format(avg_video_speed))

    # 绘制带宽和延迟图像
    # plot_metrics(sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation)

    return avg_sensor_delay,  avg_video_speed, sensor_bandwidth, video_bandwidth, sensor_delay, video_delay, sensor_speed, video_speed, sensor_allocation, video_allocation

# if __name__ == '__main__':
#     fen(1, 15)

'''
流程：

用户输入传感器抽象量x1和视频抽象量x2

初始化20个无人机,为每个无人机指定索引

根据传感器抽象量x1和视频抽象量x2计算服务需求向量

计算总的子载波数量,以及100MHz的总带宽

使用QoS模型计算资源分配

根据资源分配为每个无人机创建传感器和视频网络切片,并与相应的无人机连接

计算每个无人机的:

传感器带宽
视频带宽
传感器延迟
视频延迟
传感器速率
视频速率
计算平均传感器延迟和平均视频速率

绘制带宽和延迟图表

返回平均传感器延迟和平均视频速率
'''