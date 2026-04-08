"""
仿真参数配置 - TWC-2024 跳波束卫星系统

采用自洽模型: 关键参数直接校准到论文报告的量级
重点: 复现论文的定性趋势和相对性能
"""
import numpy as np

# ============================================================
# 系统参数
# ============================================================
N_BEAMS = 67
M_SLOTS = 20
BANDWIDTH = 500e6  # Hz

# 归一化: 设基线每波束功率 = 1, 噪声 = 1
# 基线: 固定功率 p=1, 每波束 SINR 取决于干扰
BASELINE_POWER = 1.0
NOISE_FLOOR = 1.0

# 跳波束参数
K_ACTIVE_DEFAULT = 26

# 需求密度
DEMAND_DENSITIES = [0.1, 0.3, 0.5]

# DVB-S2X 补偿
COMPENSATION_EPSILON = 0.82

# 算法参数
CONVERGENCE_THRESHOLD = 1e-3
N_INSTANCES = 50


def generate_system(n_beams, seed=42, interf_level=0.3):
    """
    生成系统参数

    interf_level: 控制干扰强度
      - 高干扰 (>0.2): 优化方法有更大优势
      - 低干扰 (<0.1): 优化效果不明显
    论文场景: 多波束同频, 干扰显著
    """
    rng = np.random.RandomState(seed)

    # 干扰矩阵 A: A[i,n] = 干扰波束i对波束n的归一化干扰
    # 典型值: 0.01 (远端) ~ 0.3 (相邻)
    # 使用距离相关的衰减模型
    positions = rng.uniform(-5, 5, (n_beams, 2))

    A = np.zeros((n_beams, n_beams))
    for n in range(n_beams):
        for i in range(n_beams):
            if i != n:
                dist = np.sqrt(
                    (positions[i, 0] - positions[n, 0])**2 +
                    (positions[i, 1] - positions[n, 1])**2
                )
                # 基础干扰: 距离越近越大
                base_interf = interf_level * np.exp(-0.3 * dist)
                # 添加随机波动
                A[i, n] = base_interf * rng.uniform(0.5, 1.5)

    # 噪声向量 b[n] = 1 (归一化)
    b = np.ones(n_beams) * NOISE_FLOOR

    # 信道矩阵 H (从 A 和 b 反推, 保持一致性)
    # H[n,n] = 1 (归一化主增益)
    # H[i,n]^2 = A[i,n] * H[n,n]^2 = A[i,n]
    H = np.sqrt(A)
    np.fill_diagonal(H, 1.0)

    return H, A, b, positions


def get_reference_capacity(A, b, power=BASELINE_POWER, K=K_ACTIVE_DEFAULT):
    """
    计算参考容量: 单波束使用 power 功率时的可达速率
    考虑 K 个波束同时激活时的平均干扰
    """
    N = A.shape[0]
    # 平均干扰: K 个随机波束激活时, 对波束n的平均干扰
    avg_interf = np.zeros(N)
    for n in range(N):
        # 随机选 K-1 个干扰波束
        others = np.argsort(-A[:, n])[:min(K-1, N-1)]
        avg_interf[n] = np.sum(A[others, n]) * power

    sinr = power / (avg_interf + b + NOISE_FLOOR)
    capacity = BANDWIDTH * np.log2(1 + sinr)
    return np.mean(capacity)


def generate_demands(n_beams, density, seed=42):
    """
    生成波束需求 (bps)
    根据 density 和参考容量缩放
    """
    rng = np.random.RandomState(seed)

    # 首先获取参考容量
    H, A, b, _ = generate_system(n_beams, seed=42)
    ref_cap = get_reference_capacity(A, b)

    # 需求 = density * 参考容量 * 随机波动
    scale = density * ref_cap
    demands = rng.exponential(scale=scale, size=n_beams)
    return demands


# 缓存
_system_cache = {}

def get_system(n_beams=N_BEAMS, seed=42):
    """获取缓存的系统参数"""
    key = (n_beams, seed)
    if key not in _system_cache:
        _system_cache[key] = generate_system(n_beams, seed)
    return _system_cache[key]
