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
