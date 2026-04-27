"""
信道模型: 卫星网络拓扑与信道系数
基于论文 Section II 系统模型, 采用归一化信道模型保证 SINR 在合理范围
"""

import numpy as np
from config import *


class SatelliteNetwork:
    """多卫星跳波束网络拓扑与信道模型"""

    def __init__(self, num_sats=NUM_SATELLITES, num_cells=NUM_CELLS,
                 num_beams=NUM_BEAMS, num_freq=NUM_FREQ_SEGMENTS, seed=42):
        self.S = num_sats
        self.K = num_cells
        self.Nb = num_beams
        self.L = num_freq
        self.rng = np.random.RandomState(seed)

        self._generate_topology()
        self._compute_coverage()
        self._setup_gso_stations()

    def _generate_topology(self):
        """生成卫星和小区的几何位置"""
        # 卫星位置: 沿轨道均匀分布, 间距 207 km
        self.sat_positions = np.zeros((self.S, 2))  # (x, y) 投影坐标 (km)
        for s in range(self.S):
            offset = (s - (self.S - 1) / 2) * SAT_DISTANCE / 1e3
            self.sat_positions[s] = [offset, 0]

        # 小区位置: 在覆盖区域内网格分布
        coverage_radius = 350  # km
        cols = int(np.ceil(np.sqrt(self.K * 1.5)))
        rows = int(np.ceil(self.K / cols))
        self.cell_positions = np.zeros((self.K, 2))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= self.K:
                    break
                x = (j - cols / 2 + 0.5) * coverage_radius * 2 / cols
                y = (i - rows / 2 + 0.5) * coverage_radius * 2 / rows * 0.9
                self.cell_positions[idx] = [x, y]
                idx += 1

        # 距离矩阵 (km) - 3D 斜距 和 地面距离
        alt_km = ALTITUDE / 1e3  # 轨道高度 (km)
        self.dist_sk = np.zeros((self.S, self.K))       # 3D 斜距 (用于信道衰减)
        self.ground_dist_sk = np.zeros((self.S, self.K))  # 地面距离 (用于覆盖判断)
        for s in range(self.S):
            for k in range(self.K):
                ground_dist = np.linalg.norm(
                    self.sat_positions[s] - self.cell_positions[k])
                self.ground_dist_sk[s, k] = ground_dist
                # 3D 斜距 = sqrt(地面距离^2 + 高度^2)
                self.dist_sk[s, k] = np.sqrt(ground_dist**2 + alt_km**2)

        # 离轴角 (度): beam 指向 cell_j 时, cell_k 相对波束中心的偏角
        # 简化: 用地面投影角近似
        self.off_axis_deg = np.zeros((self.S, self.K, self.K))
        for s in range(self.S):
            for j in range(self.K):
                for k in range(self.K):
                    dx = self.cell_positions[k][0] - self.cell_positions[j][0]
                    dy = self.cell_positions[k][1] - self.cell_positions[j][1]
                    ground_dist = np.sqrt(dx**2 + dy**2)  # km
                    # 离轴角 ≈ arctan(ground_dist / altitude)
                    self.off_axis_deg[s, j, k] = np.degrees(
                        np.arctan2(ground_dist, ALTITUDE / 1e3))

    def _compute_coverage(self):
        """计算覆盖关系 — 基于地面距离而非3D斜距"""
        coverage_radius = 300  # km, 地面覆盖半径

        self.omega = {}
        for s in range(self.S):
            self.omega[s] = [k for k in range(self.K)
                             if self.ground_dist_sk[s, k] < coverage_radius]

        self.phi = {}
        for k in range(self.K):
            self.phi[k] = [s for s in range(self.S)
                           if k in self.omega[s]]
            if not self.phi[k]:
                nearest_s = np.argmin(self.ground_dist_sk[:, k])
                self.phi[k] = [nearest_s]
                if k not in self.omega[nearest_s]:
                    self.omega[nearest_s].append(k)

        self.coverage_mask = np.zeros((self.S, self.K), dtype=bool)
        for s in range(self.S):
            for k in self.omega[s]:
                self.coverage_mask[s, k] = True

    def _setup_gso_stations(self):
        """设置 GSO 地面站"""
        gso_indices = self.rng.choice(self.K,
                                       size=min(NUM_GSO_STATIONS, self.K),
                                       replace=False)
        self.k_gso = set(int(x) for x in gso_indices)
        self.l_overlap = {}
        for k in self.k_gso:
            n_overlap = min(FREQ_OVERLAP_PER_STATION, self.L)
            self.l_overlap[k] = list(self.rng.choice(self.L,
                                                       size=n_overlap,
                                                       replace=False))

    def _antenna_gain(self, off_axis_deg):
        """
        卫星发射天线增益模型 (ITU-R S.1528 简化)
        Ka 频段 LEO 卫星高增益点波束: 峰值 ~45 dBi
        返回线性增益值
        """
        theta_3db = 1.2  # 3dB 波束宽度 (度), Ka 频段点波束
        g_peak_db = 45   # 峰值增益 (dBi)
        g_side_db = -5   # 旁瓣 (dBi)

        if off_axis_deg <= theta_3db:
            gain_db = g_peak_db - 12 * (off_axis_deg / theta_3db) ** 2
        elif off_axis_deg <= 6 * theta_3db:
            gain_db = g_peak_db - 12 - 25 * np.log10(off_axis_deg / theta_3db)
        else:
            gain_db = g_side_db

        return 10 ** (gain_db / 10)

    def generate_channel_coefficients(self, time_slot=0):
        """
        生成信道系数 h_{s,k,j}[n] 和 GSO 干扰信道 g_{s,k,j}[n]
        采用归一化模型使 SINR 在关键范围 (0-15 dB), 确保功率分配有实际影响
        """
        rng = np.random.RandomState(seed=time_slot * 1000 + 7)
        S, K, L = self.S, self.K, self.L

        # 目标: 在中等功率 (P_MAX/4/L ≈ 3.75W per beam per freq) 下
        # 主波束 SINR ≈ 5-10 dB, 干扰使 SINR 下降 3-6 dB
        # 这样功率分配才有意义 (增加功率→增加速率, 但也增加干扰)

        h = np.zeros((S, K, K))

        # 校准: 使 20Gbps 需求需要约 25W 发射功率 (5W 电路 + 25W 发射 = 30W 总)
        # 每波束功率 = 25W / (4×4) ≈ 1.56W per (beam,freq)
        # 设 SINR ≈ 5 dB (3.16 linear) 在此功率下
        # h = SINR × (N0W + I) / P, I ≈ 2×N0W
        p_ref = 25.0 / (self.Nb * L)
        target_sinr = 10 ** (5 / 10)  # 5 dB
        noise_eff = NOISE_POWER * 3  # 含干扰等效噪声
        h_main = target_sinr * noise_eff / p_ref

        for s in range(S):
            for j in range(K):
                if not self.coverage_mask[s, j]:
                    continue
                for k in range(K):
                    off_axis = self.off_axis_deg[s, j, k]
                    if k == j:
                        # 主波束: h_main + 距离衰减 + 衰落
                        d_factor = (1200 / max(self.dist_sk[s, k], 1100)) ** 2
                        fading = rng.exponential(0.6) + 0.4
                        h[s, k, j] = h_main * d_factor * fading
                    else:
                        # 干扰: 比主波束低 15-30 dB (取决于离轴角)
                        if off_axis < 2:
                            att = 0.03  # 近距离强干扰 ~-15 dB
                        elif off_axis < 5:
                            att = 0.005  # 中距离 ~-23 dB
                        elif off_axis < 10:
                            att = 5e-4  # 远距离 ~-33 dB
                        else:
                            att = 1e-4  # 极远 ~-40 dB
                        d_factor = (1200 / max(self.dist_sk[s, k], 1100)) ** 2
                        fading = rng.exponential(0.3) + 0.2
                        h[s, k, j] = h_main * att * d_factor * fading

        # GSO 干扰信道: 需要足够大使干扰约束有实际影响
        # Z_max 范围: -140 ~ -120 dBW = 1e-14 ~ 1e-12 W
        # 每个波束在重叠频段的功率 ~1W → g 需要使 Σg·P 落在此范围
        g = np.zeros((S, K, K))
        for s in range(S):
            for j in range(K):
                if not self.coverage_mask[s, j]:
                    continue
                for k in self.k_gso:
                    off_axis = self.off_axis_deg[s, j, k]
                    # GSO 地面站高增益天线: 干扰信道比主波束低约 20dB
                    att = 0.01 if off_axis < 5 else 0.002
                    g[s, k, j] = h_main * att

        return h, g

    def generate_arrivals(self, demand_total=DEMAND_DEFAULT, time_slot=0):
        """
        生成分组到达数 a_{s,k}[n]
        """
        rng = np.random.RandomState(seed=time_slot * 777 + 13)

        base_seed = (time_slot // DRIFT_INTERVAL) * 999
        rng_base = np.random.RandomState(seed=base_seed + 42)
        base_rates = rng_base.uniform(ARRIVAL_RATE_MIN, ARRIVAL_RATE_MAX, self.K)

        total_base = np.sum(base_rates)
        if total_base > 0:
            scale = demand_total / total_base
            rates = base_rates * scale
        else:
            rates = base_rates

        a = np.zeros((self.S, self.K))
        for s in range(self.S):
            for k in self.omega[s]:
                avg_packets = rates[k] * T0 / M0
                avg_packets = max(avg_packets, 0.1)
                a[s, k] = rng.poisson(avg_packets)

        return a, rates
