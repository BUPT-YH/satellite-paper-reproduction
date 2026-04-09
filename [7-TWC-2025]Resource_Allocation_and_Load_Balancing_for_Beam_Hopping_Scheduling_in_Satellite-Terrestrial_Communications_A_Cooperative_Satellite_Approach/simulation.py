"""
主仿真逻辑
完整实现卫星-地面通信的跳波束资源分配与负载均衡仿真
"""

import numpy as np
import sys
import os

import config as cfg
from beam_hopping_drl import DQNAgent, GreedyBHScheduler, USWGScheduler, PreSchedulingBH
from resource_allocation import (generate_channel_gains, mm_resource_allocation,
                                  compute_throughput_per_beam)
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

        # 生成小区到卫星的覆盖关系
        self._generate_coverage()

        # 初始化业务速率
        self._init_traffic(total_traffic_gbps)

        # 初始化缓冲区状态 D_s(n): 每颗卫星 (n_cells_per_sat, Tth)
        self.buffers = [np.zeros((self.n_cells_per_sat, cfg.Tth)) for _ in range(n_sat)]

        # 生成信道增益
        # 简化: 小区到卫星的距离 (基于轨道高度和偏移)
        self.cell_distances = np.random.uniform(cfg.altitude * 0.95, cfg.altitude * 1.1,
                                                 size=n_beams)
        self.h = generate_channel_gains(n_sat, self.n_cells_per_sat, n_beams, self.cell_distances)

        # 负载均衡
        self.lambda_si = None
        self.throughput_history = []

    def _generate_coverage(self):
        """生成小区覆盖关系"""
        # 每颗卫星覆盖 n_cells_per_sat 个小区
        # 使用简化的重叠覆盖模型
        self.cell_to_sats = {}  # cell_idx -> [sat_idx, ...]
        self.sat_to_cells = {}  # sat_idx -> [cell_idx, ...]

        for s in range(self.n_sat):
            start = (s * self.n_cells_per_sat // 2) % self.n_cells
            cells = [(start + i) % self.n_cells for i in range(self.n_cells_per_sat)]
            self.sat_to_cells[s] = cells
            for c in cells:
                if c not in self.cell_to_sats:
                    self.cell_to_sats[c] = []
                self.cell_to_sats[c].append(s)

    def _init_traffic(self, total_gbps):
        """初始化各小区业务速率"""
        # 各小区速率在 [traffic_min, traffic_max] Mbps 范围内随机
        n = self.n_cells
        rates = np.random.uniform(cfg.traffic_min, cfg.traffic_max, size=n)
        # 调整总和以匹配总业务量
        current_total = rates.sum()
        if current_total > 0:
            target_total = total_gbps * 1000  # Gbps -> Mbps
            rates = rates / current_total * target_total
            rates = np.clip(rates, cfg.traffic_min, cfg.traffic_max)
        self.lambda_arrival = rates  # Mbps

    def update_traffic(self, slot):
        """每 traffic_walk_interval 时隙更新业务 (随机游走)"""
        if slot > 0 and slot % cfg.traffic_walk_interval == 0:
            walk = np.random.uniform(-cfg.traffic_walk_range, cfg.traffic_walk_range,
                                      size=self.n_cells)
            self.lambda_arrival = np.clip(self.lambda_arrival + walk,
                                           cfg.traffic_min, cfg.traffic_max)

    def update_buffer_arrival(self):
        """新数据包到达缓冲区"""
        for s in range(self.n_sat):
            cells = self.sat_to_cells[s]
            for local_i in range(self.n_cells_per_sat):
                cell_idx = cells[local_i] if local_i < len(cells) else local_i
                # 泊松到达 (简化为高斯近似)
                arrival_rate = self.lambda_arrival[cell_idx % self.n_cells] / 8  # 转换为包数
                new_packets = max(0, np.random.poisson(arrival_rate))
                # 新到达的数据放在 ψ^1 (第0列)
                self.buffers[s][local_i, 0] += new_packets

    def age_buffer(self):
        """缓冲区数据老化 (时隙推进)"""
        for s in range(self.n_sat):
            # 数据包等待时间+1, 超过Tth的丢弃
            self.buffers[s][:, 1:cfg.Tth] = self.buffers[s][:, 0:cfg.Tth - 1]
            self.buffers[s][:, 0] = 0  # 新到达单独处理

    def compute_queuing_delay(self, s, local_i):
        """计算排队延迟 τ_{s,i}(n) (公式4)"""
        queue = self.buffers[s][local_i]
        total = queue[1:].sum()
        if total == 0:
            return 0
        weighted = sum((t) * queue[t] for t in range(1, cfg.Tth))
        return weighted / total

    def compute_latency_metric(self, s):
        """计算延迟度量 Δτ_s(n) (公式11)"""
        delays = []
        for i in range(self.n_cells_per_sat):
            delays.append(self.compute_queuing_delay(s, i))

        tau_max = max(delays) if delays else 0
        tau_min = min(delays) if delays else 0
        delta_tau = (tau_max - tau_min) + cfg.kappa * tau_max
        return delta_tau, tau_max, tau_min

    def compute_reward(self, s, throughputs):
        """计算奖励 r_s(n) (公式12)"""
        total_throughput = throughputs[s].sum()
        delta_tau, _, _ = self.compute_latency_metric(s)
        reward = self.beta * total_throughput / cfg.Y0 - (1 - self.beta) * delta_tau / cfg.Gamma0
        return reward, total_throughput, delta_tau

    def transmit_data(self, s, bh_pattern, throughputs):
        """从缓冲区传输数据"""
        for k, cell_local in enumerate(bh_pattern):
            if cell_local < self.n_cells_per_sat:
                # 传输量受限于吞吐量和缓冲区数据量
                transmit = min(throughputs[s, k], self.buffers[s][cell_local].sum())
                # 从最老的数据开始传输
                remaining = transmit
                for t in range(cfg.Tth - 1, -1, -1):
                    if remaining <= 0:
                        break
                    sent = min(remaining, self.buffers[s][cell_local, t])
                    self.buffers[s][cell_local, t] -= sent
                    remaining -= sent


def run_simulation(method='proposed', n_sat=cfg.Ns, n_cells=cfg.Nc,
                    n_beams=cfg.Nb, n_cells_per_sat=cfg.omega_s,
                    beta=cfg.beta, total_traffic_gbps=35.0,
                    n_slots=2000, train_mode=False, seed=42,
                    use_drl=True, use_ra=True, use_lb=True):
    """
    运行完整仿真
    method: 'proposed', 'without_drl', 'without_ra', 'without_lb', 'original',
            'drl_avoid', 'max_uswg', 'pre_scheduling'
    """
    np.random.seed(seed)

    # 创建网络环境
    net = SatelliteNetwork(n_sat, n_cells, n_beams, cfg.NL,
                            n_cells_per_sat, beta, total_traffic_gbps, seed)

    # 创建调度器
    agents = []
    for s in range(n_sat):
        if method == 'proposed' or (method == 'custom' and use_drl):
            agent = DQNAgent(n_cells_per_sat, n_beams)
            agents.append(agent)
        elif method == 'max_uswg':
            agents.append(USWGScheduler(n_cells_per_sat, n_beams))
        elif method == 'pre_scheduling':
            agents.append(PreSchedulingBH(n_cells_per_sat, n_beams))
        elif method == 'drl_avoid':
            agent = DQNAgent(n_cells_per_sat, n_beams)
            agents.append(agent)
        else:
            agents.append(GreedyBHScheduler(n_cells_per_sat, n_beams))

    # 如果是DRL方法，先训练
    if method in ('proposed', 'drl_avoid') or (method == 'custom' and use_drl):
        train_network(net, agents, n_sat, n_beams, n_cells_per_sat,
                      beta, use_ra, use_lb, seed)

    # 评估阶段
    rewards_history = []
    throughput_history = []
    latency_history = []
    total_throughputs = []
    total_latencies = []
    training_rewards = []  # 记录训练期间的reward

    np.random.seed(seed + 1000)

    for slot in range(n_slots):
        # 更新业务
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
            state = net.buffers[s]
            if isinstance(agents[s], (DQNAgent,)):
                agents[s].epsilon = 0.05  # 评估阶段低探索率
                action = agents[s].select_action(state)
            else:
                action = agents[s].select_action(state)
            bh_patterns.append(action)

        # 资源分配
        if use_ra and method != 'original':
            f_alloc, Pb = mm_resource_allocation(
                bh_patterns, net.h, n_sat, n_beams, cfg.NL,
                cfg.P_sat, cfg.P_tot)
        else:
            # 无资源分配: 所有波束使用所有频率段
            f_alloc = np.ones((n_sat, n_beams, cfg.NL))
            Pb = cfg.P_sat / (n_beams * cfg.NL)

        # 计算吞吐量
        throughputs = compute_throughput_per_beam(f_alloc, Pb, net.h, n_sat, n_beams, cfg.NL)

        # 传输数据
        for s in range(n_sat):
            net.transmit_data(s, bh_patterns[s], throughputs)

        # 计算奖励和指标
        slot_throughput = 0
        slot_latency = 0
        for s in range(n_sat):
            reward, tp, lt = net.compute_reward(s, throughputs)
            slot_throughput += tp
            slot_latency += lt
            rewards_history.append(reward)
            training_rewards.append(reward)

        total_throughputs.append(slot_throughput / n_cells)  # 每小区平均
        total_latencies.append(slot_latency / n_sat)
        net.throughput_history.append(throughputs)

    avg_throughput = np.mean(total_throughputs[-1000:])  # 取后半段平均
    avg_latency = np.mean(total_latencies[-1000:])

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

    for slot in range(cfg.train_slots):
        net.update_traffic(slot)
        net.age_buffer()
        net.update_buffer_arrival()

        # 负载均衡
        if use_lb and slot % cfg.Tb == 0:
            net.lambda_si = run_load_balancing(
                net.lambda_arrival, net.throughput_history,
                n_sat, net.n_cells, n_cells_per_sat,
                list(net.sat_to_cells.values()), n_beams)

        # 选择动作
        bh_patterns = []
        for s in range(n_sat):
            state = net.buffers[s]
            action = agents[s].select_action(state)
            bh_patterns.append(action)

        # 资源分配
        if use_ra:
            f_alloc, Pb = mm_resource_allocation(
                bh_patterns, net.h, n_sat, n_beams, cfg.NL,
                cfg.P_sat, cfg.P_tot)
        else:
            f_alloc = np.ones((n_sat, n_beams, cfg.NL))
            Pb = cfg.P_sat / (n_beams * cfg.NL)

        throughputs = compute_throughput_per_beam(f_alloc, Pb, net.h, n_sat, n_beams, cfg.NL)

        # 传输
        for s in range(n_sat):
            net.transmit_data(s, bh_patterns[s], throughputs)

        # 计算奖励并存储经验
        for s in range(n_sat):
            reward, _, _ = net.compute_reward(s, throughputs)
            old_state = net.buffers[s].copy()
            agents[s].store_experience(old_state, bh_patterns[s], reward, net.buffers[s])

            # 学习
            if slot % 4 == 0:
                agents[s].update()

            # 更新目标网络
            if slot % cfg.Cu == 0:
                agents[s].update_target_network()

        net.throughput_history.append(throughputs)

        if slot % 1000 == 0:
            print(f"  Training slot {slot}/{cfg.train_slots}")
