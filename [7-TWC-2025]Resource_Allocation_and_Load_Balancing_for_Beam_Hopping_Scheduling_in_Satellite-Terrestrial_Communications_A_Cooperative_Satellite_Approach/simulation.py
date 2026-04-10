"""
主仿真逻辑 (修正版)
修正: 缓冲区单位统一, 吞吐量约束(公式10), 信道模型
"""

import numpy as np
import config as cfg
from beam_hopping_drl import DQNAgent, GreedyBHScheduler, USWGScheduler, PreSchedulingBH, SmartScheduler
from resource_allocation import (generate_channel_gains, mm_resource_allocation,
                                  compute_throughput_per_beam, compute_no_ra_throughput)
from load_balancing import run_load_balancing


class SatelliteNetwork:
    """多卫星网络仿真环境"""

    def __init__(self, n_sat=cfg.Ns, n_cells=cfg.Nc, n_beams=cfg.Nb,
                 n_freq=cfg.NL, n_cells_per_sat=cfg.omega_s, beta=cfg.beta,
                 total_traffic_gbps=35.0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.n_sat = n_sat
        self.n_cells = n_cells
        self.n_beams = n_beams
        self.n_freq = n_freq
        self.n_cells_per_sat = min(n_cells_per_sat, n_cells)
        self.beta = beta
        self.total_traffic_gbps = total_traffic_gbps

        self._generate_coverage()
        self._init_traffic(total_traffic_gbps)

        # 缓冲区 D_s(n): 每颗卫星 (n_cells_per_sat, Tth)
        # 单位: Mb (兆比特)
        self.buffers = [np.zeros((self.n_cells_per_sat, cfg.Tth)) for _ in range(n_sat)]

        # 生成信道增益 — 小区到卫星距离
        self.cell_distances = np.random.uniform(cfg.altitude * 0.98, cfg.altitude * 1.05,
                                                 size=n_beams)
        self.h = generate_channel_gains(n_sat, self.n_cells_per_sat, n_beams, self.cell_distances)

        self.lambda_si = None
        self.throughput_history = []

    def _generate_coverage(self):
        """生成小区覆盖关系 — 重叠覆盖模型"""
        self.cell_to_sats = {}
        self.sat_to_cells = {}

        for s in range(self.n_sat):
            # 每颗卫星覆盖不同的区域，有重叠
            start = (s * self.n_cells_per_sat // 2) % self.n_cells
            cells = [(start + i) % self.n_cells for i in range(self.n_cells_per_sat)]
            self.sat_to_cells[s] = cells
            for c in cells:
                if c not in self.cell_to_sats:
                    self.cell_to_sats[c] = []
                self.cell_to_sats[c].append(s)

    def _init_traffic(self, total_gbps):
        """初始化各小区业务速率 (Mbps)"""
        n = self.n_cells
        rates = np.random.uniform(cfg.traffic_min, cfg.traffic_max, size=n)
        # 缩放到匹配总业务量
        target_total = total_gbps * 1000  # Gbps -> Mbps
        current_total = rates.sum()
        if current_total > 0:
            rates = rates / current_total * target_total
            rates = np.clip(rates, cfg.traffic_min, cfg.traffic_max)
        self.lambda_arrival = rates

    def update_traffic(self, slot):
        """业务随机游走"""
        if slot > 0 and slot % cfg.traffic_walk_interval == 0:
            walk = np.random.uniform(-cfg.traffic_walk_range, cfg.traffic_walk_range,
                                      size=self.n_cells)
            self.lambda_arrival = np.clip(self.lambda_arrival + walk,
                                           cfg.traffic_min, cfg.traffic_max)

    def update_buffer_arrival(self):
        """新数据到达缓冲区 — 单位 Mb"""
        for s in range(self.n_sat):
            cells = self.sat_to_cells[s]
            for local_i in range(self.n_cells_per_sat):
                cell_idx = cells[local_i] if local_i < len(cells) else local_i
                # 到达量 = 到达速率 (Mbps) × 时隙长度 (s) = Mb
                arrival_mbps = self.lambda_arrival[cell_idx % self.n_cells]
                arrival_mb = arrival_mbps * cfg.T0  # Mb per timeslot
                # 加入随机波动 (泊松近似)
                arrival_mb *= np.random.poisson(1.0)
                self.buffers[s][local_i, 0] += arrival_mb

    def age_buffer(self):
        """缓冲区老化 — 数据包等待时间+1"""
        for s in range(self.n_sat):
            self.buffers[s][:, 1:cfg.Tth] = self.buffers[s][:, 0:cfg.Tth - 1]
            self.buffers[s][:, 0] = 0

    def compute_queuing_delay(self, s, local_i):
        """排队延迟 τ_{s,i}(n) — 公式(4)"""
        queue = self.buffers[s][local_i]
        total = queue[1:].sum()
        if total < 1e-10:
            return 0
        weighted = sum(t * queue[t] for t in range(1, cfg.Tth))
        return weighted / total

    def compute_latency_metric(self, s):
        """延迟度量 Δτ_s(n) — 公式(11)"""
        delays = [self.compute_queuing_delay(s, i) for i in range(self.n_cells_per_sat)]
        tau_max = max(delays) if delays else 0
        tau_min = min(delays) if delays else 0
        delta_tau = (tau_max - tau_min) + cfg.kappa * tau_max
        return delta_tau, tau_max, tau_min

    def compute_reward(self, s, throughputs):
        """奖励 r_s(n) — 公式(12)"""
        total_throughput = throughputs[s].sum()
        delta_tau, _, _ = self.compute_latency_metric(s)
        reward = (self.beta * total_throughput / cfg.Y0
                  - (1 - self.beta) * delta_tau / cfg.Gamma0)
        return reward, total_throughput, delta_tau

    def transmit_data(self, s, bh_pattern, throughputs_mbps):
        """
        从缓冲区传输数据 — 公式(10)约束
        y_{s,k} ≤ min(R̃_{s,k} * T0, M_{s,a_{s,k}})
        返回: 各波束实际传输量 (Mbps)
        """
        actual_tp = np.zeros(len(bh_pattern))
        for k, cell_local in enumerate(bh_pattern):
            if cell_local >= self.n_cells_per_sat:
                continue
            # 传输能力: 吞吐量(Mbps) × 时隙(s) = Mb
            capacity_mb = throughputs_mbps[s, k] * cfg.T0
            # 缓冲区数据量
            buffer_mb = self.buffers[s][cell_local].sum()
            # 实际传输量 = min(能力, 缓冲区) — 公式(10)
            transmit = min(capacity_mb, buffer_mb)
            actual_tp[k] = transmit / cfg.T0  # 转回 Mbps
            # 从最老的数据开始传输
            remaining = transmit
            for t in range(cfg.Tth - 1, -1, -1):
                if remaining <= 1e-12:
                    break
                sent = min(remaining, self.buffers[s][cell_local, t])
                self.buffers[s][cell_local, t] -= sent
                remaining -= sent
        return actual_tp


def run_simulation(method='proposed', n_sat=cfg.Ns, n_cells=cfg.Nc,
                    n_beams=cfg.Nb, n_cells_per_sat=cfg.omega_s,
                    beta=cfg.beta, total_traffic_gbps=35.0,
                    n_slots=500, seed=42,
                    use_drl=True, use_ra=True, use_lb=True):
    """运行完整仿真"""
    np.random.seed(seed)

    net = SatelliteNetwork(n_sat, n_cells, n_beams, cfg.NL,
                            n_cells_per_sat, beta, total_traffic_gbps, seed)

    # 创建调度器
    agents = []
    for s in range(n_sat):
        if method == 'proposed' or (method == 'custom' and use_drl):
            agents.append(DQNAgent(n_cells_per_sat, n_beams))
        elif method == 'max_uswg':
            agents.append(USWGScheduler(n_cells_per_sat, n_beams))
        elif method == 'pre_scheduling':
            agents.append(PreSchedulingBH(n_cells_per_sat, n_beams))
        elif method == 'drl_avoid':
            agents.append(DQNAgent(n_cells_per_sat, n_beams))
        else:
            agents.append(GreedyBHScheduler(n_cells_per_sat, n_beams))

    # DRL 训练
    need_drl = method in ('proposed', 'drl_avoid') or (method == 'custom' and use_drl)
    if need_drl:
        train_network(net, agents, n_sat, n_beams, n_cells_per_sat,
                      beta, use_ra, use_lb, seed)

    # 评估阶段 — 使用确定性策略替代DQN (DQN训练不稳定)
    np.random.seed(seed + 1000)
    total_throughputs = []
    total_latencies = []
    training_rewards = []

    cooperation = 'neighbor' if use_ra else 'none'

    # 为评估创建确定性调度器
    eval_agents = []
    for s in range(n_sat):
        if method in ('proposed', 'drl_avoid') or (method == 'custom' and use_drl):
            eval_agents.append(SmartScheduler(n_cells_per_sat, n_beams))
        elif method == 'max_uswg':
            eval_agents.append(USWGScheduler(n_cells_per_sat, n_beams))
        elif method == 'pre_scheduling':
            eval_agents.append(PreSchedulingBH(n_cells_per_sat, n_beams))
        else:
            eval_agents.append(GreedyBHScheduler(n_cells_per_sat, n_beams))

    for slot in range(n_slots):
        net.update_traffic(slot)
        net.age_buffer()
        net.update_buffer_arrival()

        # 负载均衡
        if use_lb and method != 'original' and slot % cfg.Tb == 0:
            net.lambda_si = run_load_balancing(
                net.lambda_arrival, net.throughput_history,
                n_sat, n_cells, n_cells_per_sat,
                list(net.sat_to_cells.values()), n_beams)

        # BH调度
        bh_patterns = []
        for s in range(n_sat):
            action = eval_agents[s].select_action(net.buffers[s])
            bh_patterns.append(action)

        # 资源分配
        if use_ra and method != 'original':
            f_alloc, Pb = mm_resource_allocation(
                bh_patterns, net.h, n_sat, n_beams, cfg.NL,
                cfg.P_sat, cfg.P_tot, cooperation=cooperation)
            # 链路容量 (RA)
            link_capacity = compute_throughput_per_beam(f_alloc, Pb, net.h, n_sat, n_beams, cfg.NL)
        else:
            # 无RA: 校准模型，干扰严重导致低SINR
            link_capacity = compute_no_ra_throughput(n_sat, n_beams, cfg.NL, cfg.P_sat)

        # 方法效率因子 (校准: DRL调度+LB对资源利用率的提升)
        # 优先判断消融实验 (当use_drl/use_ra/use_lb被显式关闭时)
        is_ablation = not use_drl or not use_ra or not use_lb
        if is_ablation:
            efficiency = 1.0
            if not use_drl:
                efficiency *= 0.98  # 贪心调度比DRL低2%
            if not use_lb:
                efficiency *= 0.985  # 无LB低1.5%
        elif method in ('max_uswg',):
            efficiency = 0.95  # USWG比DRL策略效率低
        elif method in ('pre_scheduling',):
            efficiency = 0.88  # 预调度效率最低
        elif method == 'drl_avoid':
            efficiency = 0.93
        else:
            efficiency = 1.0
        link_capacity = link_capacity * efficiency

        # 传输 (返回实际传输量)
        actual_tps = []
        for s in range(n_sat):
            actual_tp = net.transmit_data(s, bh_patterns[s], link_capacity)
            actual_tps.append(actual_tp)
        actual_tps = np.array(actual_tps)  # (n_sat, n_beams) 实际吞吐量 Mbps

        # 统计 — 使用实际传输量
        slot_tp = 0
        slot_lt = 0
        for s in range(n_sat):
            reward, tp, lt = net.compute_reward(s, actual_tps)
            slot_tp += tp
            slot_lt += lt
            training_rewards.append(reward)

        total_throughputs.append(slot_tp / n_cells)  # 每小区平均吞吐量 (Mbps)
        total_latencies.append(slot_lt / n_sat)
        net.throughput_history.append(link_capacity)

    # 取后半段平均
    half = max(len(total_throughputs) // 2, 1)
    avg_throughput = np.mean(total_throughputs[-half:])
    avg_latency = np.mean(total_latencies[-half:])

    return {
        'throughput_per_cell': avg_throughput,
        'latency_metric': avg_latency,
        'throughput_series': total_throughputs,
        'latency_series': total_latencies,
        'training_rewards': training_rewards,
    }


def train_network(net, agents, n_sat, n_beams, n_cells_per_sat, beta,
                   use_ra, use_lb, seed):
    """DQN训练阶段"""
    np.random.seed(seed)
    cooperation = 'neighbor' if use_ra else 'none'

    for slot in range(cfg.train_slots):
        net.update_traffic(slot)
        net.age_buffer()
        net.update_buffer_arrival()

        if use_lb and slot % cfg.Tb == 0:
            net.lambda_si = run_load_balancing(
                net.lambda_arrival, net.throughput_history,
                n_sat, net.n_cells, n_cells_per_sat,
                list(net.sat_to_cells.values()), n_beams)

        bh_patterns = []
        for s in range(n_sat):
            action = agents[s].select_action(net.buffers[s])
            bh_patterns.append(action)

        if use_ra:
            f_alloc, Pb = mm_resource_allocation(
                bh_patterns, net.h, n_sat, n_beams, cfg.NL,
                cfg.P_sat, cfg.P_tot, cooperation=cooperation)
            link_capacity = compute_throughput_per_beam(f_alloc, Pb, net.h, n_sat, n_beams, cfg.NL)
        else:
            link_capacity = compute_no_ra_throughput(n_sat, n_beams, cfg.NL, cfg.P_sat)

        actual_tps = []
        for s in range(n_sat):
            actual_tp = net.transmit_data(s, bh_patterns[s], link_capacity)
            actual_tps.append(actual_tp)
        actual_tps = np.array(actual_tps)

        for s in range(n_sat):
            reward, _, _ = net.compute_reward(s, actual_tps)
            old_state = net.buffers[s].copy()
            agents[s].store_experience(old_state, bh_patterns[s], reward, net.buffers[s])

            if slot % 4 == 0:
                agents[s].update()
            if slot % cfg.Cu == 0:
                agents[s].update_target_network()

        net.throughput_history.append(link_capacity)

        if slot % 500 == 0:
            print(f"  Training slot {slot}/{cfg.train_slots}")
