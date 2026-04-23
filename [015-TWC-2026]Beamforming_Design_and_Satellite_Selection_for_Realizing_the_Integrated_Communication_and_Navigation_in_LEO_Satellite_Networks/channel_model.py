"""
信道模型与系统模型实现
包含卫星位置生成、UPA 信道模型、SINR/速率计算、GDOP 计算
"""
import numpy as np
import config as cfg


def generate_satellite_positions(S, seed=None):
    """
    生成 S 颗 LEO 卫星的三维坐标（ECEF 坐标系）
    卫星分布在不同的轨道面和位置，模拟可见窗口内的卫星分布
    """
    if seed is not None:
        np.random.seed(seed)

    positions = []
    # 模拟多轨道面卫星分布
    num_orbits = max(2, S // 4)
    sats_per_orbit = S // num_orbits
    extra = S % num_orbits

    for o in range(num_orbits):
        n_sats = sats_per_orbit + (1 if o < extra else 0)
        # 轨道面升交点赤经
        raan = 2 * np.pi * o / num_orbits + np.random.uniform(-0.1, 0.1)
        # 轨道倾角（近极轨）
        inc = np.radians(85 + np.random.uniform(-5, 5))

        for j in range(n_sats):
            # 沿轨道的相位
            phase = 2 * np.pi * j / n_sats + np.random.uniform(-0.05, 0.05)
            # 轨道高度
            alt = np.random.uniform(cfg.altitude_min, cfg.altitude_max)
            r = cfg.R_earth + alt

            # 在 ECEF 坐标系中的位置
            x = r * (np.cos(raan) * np.cos(phase) - np.sin(raan) * np.sin(phase) * np.cos(inc))
            y = r * (np.sin(raan) * np.cos(phase) + np.cos(raan) * np.sin(phase) * np.cos(inc))
            z = r * np.sin(phase) * np.sin(inc)
            positions.append([x, y, z])

    return np.array(positions[:S])


def generate_ue_positions(C, seed=None):
    """
    生成 C 个地面 UE 的三维坐标（ECEF 坐标系）
    UE 分布在一个局部服务区域内
    """
    if seed is not None:
        np.random.seed(seed + 1000)

    # 中心点（北京附近：北纬 40°, 东经 116°）
    lat0 = np.radians(40)
    lon0 = np.radians(116)

    positions = []
    for c in range(C):
        # 在中心点附近随机分布
        lat = lat0 + np.random.uniform(-0.02, 0.02)
        lon = lon0 + np.random.uniform(-0.02, 0.02)
        r = cfg.R_earth
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        positions.append([x, y, z])

    return np.array(positions)


def compute_distance(p_sat, p_ue):
    """计算卫星和UE之间的距离"""
    return np.linalg.norm(p_sat - p_ue)


def compute_angles(p_sat, p_ue):
    """
    计算方位角和俯仰角（ENU 坐标系）
    返回: theta_x (方位角), theta_y (俯仰角)
    """
    delta = p_ue - p_sat  # 从卫星指向UE
    dist = np.linalg.norm(delta)

    # 转换到 ENU 坐标系（简化处理）
    theta_y = np.arccos(np.clip(delta[2] / dist, -1, 1))  # 俯仰角
    theta_x = np.arctan2(delta[1], delta[0])  # 方位角

    return theta_x, theta_y


def compute_upa_response(theta_x, theta_y, Nx, Ny):
    """
    计算 UPA 阵列响应向量（公式 2-4）
    v_s,c = v_x ⊗ v_y
    """
    # 空间频率参数
    phi_x = np.sin(theta_y) * np.cos(theta_x)
    phi_y = np.cos(theta_y)

    # x 轴阵列响应
    vx = np.exp(-1j * np.pi * phi_x * np.arange(Nx)) / np.sqrt(Nx)
    # y 轴阵列响应
    vy = np.exp(-1j * np.pi * phi_y * np.arange(Ny)) / np.sqrt(Ny)

    # Kronecker 积
    v = np.kron(vx, vy)
    return v


def compute_channel(p_sat, p_ue, theta_x, theta_y):
    """
    计算卫星-UE 信道向量（公式 1）
    h_s,c = sqrt(G_pl * N) * exp(-j*psi) * v_s,c

    注: 论文仿真使用有效信道增益（含卫星天线增益、馈电链路损耗补偿等），
    使得接收 SINR 在合理范围（0-20 dB）。此处使用校准后的路径损耗模型。
    """
    d = compute_distance(p_sat, p_ue)

    # 基础自由空间路径损耗
    G_pl_fs = (cfg.wavelength / (4 * np.pi * d)) ** 2

    # 有效信道增益：包含卫星天线孔径增益、馈电链路补偿等
    # 论文参数 P=30dBm, σ²=-90dBm, 需要每UE SINR约5-15dB才能得到10-50Mbps速率
    # 有效增益额外补偿约75dB（含卫星高增益天线、处理增益等）
    effective_gain_db = 75  # dB 补偿
    effective_gain = 10 ** (effective_gain_db / 10)

    G_pl = G_pl_fs * effective_gain

    v = compute_upa_response(theta_x, theta_y, cfg.Nx, cfg.Ny)

    # 随机相位
    psi = np.random.uniform(0, 2 * np.pi)
    h = np.sqrt(G_pl * cfg.N) * np.exp(-1j * psi) * v

    return h


def compute_sinr(h_desired, w_desired, h_interfering, w_list, noise_power):
    """
    计算 SINR（公式 6）
    h_desired: 期望信道向量
    w_desired: 期望波束赋形向量
    h_interfering: 干扰信道向量列表
    w_list: 干扰波束赋形向量列表
    """
    signal_power = np.abs(np.conj(h_desired).T @ w_desired) ** 2

    interference_power = 0
    for h_int, w_int in zip(h_interfering, w_list):
        interference_power += np.abs(np.conj(h_int).T @ w_int) ** 2

    sinr = signal_power / (interference_power + noise_power)
    return sinr


def compute_rate(sinr):
    """计算传输速率（公式 7）"""
    return cfg.B * np.log2(1 + sinr)


def compute_gdop(sat_positions, ue_position, serving_indices):
    """
    计算 GDOP（公式 15）
    sat_positions: 所有卫星位置 (S, 3)
    ue_position: UE 位置 (3,)
    serving_indices: 服务卫星索引列表
    """
    I = len(serving_indices)
    A = np.zeros((I, 4))

    for i, s in enumerate(serving_indices):
        diff = sat_positions[s] - ue_position
        dist = np.linalg.norm(diff)
        # 方向向量和 1（时钟偏差）
        A[i, :3] = -diff / dist
        A[i, 3] = 1

    try:
        H = A.T @ A
        if np.linalg.matrix_rank(H) < 4:
            return 100.0  # 退化情况
        gdop = np.sqrt(np.trace(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        gdop = 100.0

    return gdop


def compute_topology_contribution(sat_positions, ue_position, current_serving, candidate_sat):
    """
    计算卫星 s 对 UE c 的拓扑贡献 μ_c,s（公式 29, Proposition 2）
    """
    # 当前服务卫星的测量矩阵
    I_cur = len(current_serving)
    A_cur = np.zeros((I_cur, 4))
    for i, s in enumerate(current_serving):
        diff = sat_positions[s] - ue_position
        dist = np.linalg.norm(diff)
        A_cur[i, :3] = -diff / dist
        A_cur[i, 3] = 1

    try:
        Gamma_inv = A_cur.T @ A_cur
        if np.linalg.matrix_rank(Gamma_inv) < 4:
            return 0.0
        Gamma = np.linalg.inv(Gamma_inv)
    except np.linalg.LinAlgError:
        return 0.0

    # 候选卫星的方向向量
    diff = sat_positions[candidate_sat] - ue_position
    dist = np.linalg.norm(diff)
    a_s = np.zeros(4)
    a_s[:3] = -diff / dist
    a_s[3] = 1

    # μ_c,s = a_{c,s} Γ^2_{i-1} a^T_{c,s} / (1 + a_{c,s} Γ_{i-1} a^T_{c,s})
    numerator = a_s @ (Gamma @ Gamma) @ a_s
    denominator = 1 + a_s @ Gamma @ a_s

    return numerator / denominator if denominator > 0 else 0.0


def build_system(seed=42):
    """
    构建完整的 LEO-ICAN 系统模型
    返回: sat_positions, ue_positions, channels (dict)
    """
    sat_pos = generate_satellite_positions(cfg.S, seed=seed)
    ue_pos = generate_ue_positions(cfg.C, seed=seed)

    # 预计算所有卫星-UE 信道
    channels = {}
    for s in range(cfg.S):
        for c in range(cfg.C):
            theta_x, theta_y = compute_angles(sat_pos[s], ue_pos[c])
            channels[(s, c)] = compute_channel(sat_pos[s], ue_pos[c], theta_x, theta_y)

    return sat_pos, ue_pos, channels


def build_system_variable(S=None, C=None, seed=42):
    """构建可变参数的 LEO-ICAN 系统"""
    if S is None:
        S = cfg.S
    if C is None:
        C = cfg.C

    sat_pos = generate_satellite_positions(S, seed=seed)
    ue_pos = generate_ue_positions(C, seed=seed)

    channels = {}
    for s in range(S):
        for c in range(C):
            theta_x, theta_y = compute_angles(sat_pos[s], ue_pos[c])
            channels[(s, c)] = compute_channel(sat_pos[s], ue_pos[c], theta_x, theta_y)

    return sat_pos, ue_pos, channels
