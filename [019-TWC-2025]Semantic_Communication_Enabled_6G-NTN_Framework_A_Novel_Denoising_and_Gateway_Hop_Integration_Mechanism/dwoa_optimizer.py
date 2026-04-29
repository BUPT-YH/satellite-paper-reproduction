"""
离散鲸鱼优化算法 (DWOA) 实现
用于子载波分配和路径选择优化
基于论文 Algorithm 3 和公式 (36)-(43)
"""

import numpy as np
from config import *


class DWOAOptimizer:
    """离散鲸鱼优化算法 - 两阶段子载波分配"""

    def __init__(self, population_size=DWOA_POPULATION, max_iter=DWOA_MAX_ITER, seed=RANDOM_SEED):
        self.pop_size = population_size
        self.max_iter = max_iter
        self.b = DWOA_B_CONSTANT
        self.rng = np.random.RandomState(seed)

    def _init_population(self, num_nodes, num_subcarriers):
        """
        初始化种群: 每个个体是一个子载波分配方案
        染色体长度 = num_nodes, 每个基因 ∈ {0, 1, ..., num_subcarriers-1, -1}
        值 >= 0 表示分配的子载波编号, -1 表示未分配
        """
        pop = np.zeros((self.pop_size, num_nodes), dtype=int)
        for i in range(self.pop_size):
            for j in range(num_nodes):
                if self.rng.random() < 0.9:  # 90%概率分配子载波
                    pop[i, j] = self.rng.randint(0, num_subcarriers)
                else:
                    pop[i, j] = -1
        return pop

    def _discretize(self, continuous_val, num_subcarriers):
        """将连续值映射到离散子载波编号"""
        discrete = int(np.round(np.abs(continuous_val))) % num_subcarriers
        return max(0, discrete)

    def _fitness_stage1(self, solution, sat_freqs, distances, noise_w,
                        gain_receiver, bandwidth, data_length, psnr_req,
                        psnr_model):
        """
        Stage 1 适应度: 卫星→地面节点 (直接GU + 网关) 的子载波分配
        目标: 最小化平均延迟 (Eq. 24), 约束: PSNR >= Ψ
        """
        num_nodes = len(solution)
        total_latency = 0.0
        violations = 0
        used_subcarriers = set()

        for i in range(num_nodes):
            k = solution[i]
            if k < 0:
                violations += 1
                total_latency += QOS_PENALTY
                continue

            # 检查子载波冲突 (约束 27c)
            if k in used_subcarriers:
                violations += 1
                total_latency += QOS_PENALTY * 0.5
                continue
            used_subcarriers.add(k)

            # 计算SNR和速率
            snr = self._compute_snr(sat_freqs[k], distances[i],
                                    gain_receiver, noise_w[i])
            rate = bandwidth * np.log2(1 + max(snr, 1e-10))
            if rate <= 0:
                violations += 1
                total_latency += QOS_PENALTY
                continue

            latency = data_length / rate
            total_latency += latency

            # 检查PSNR约束
            snr_db = 10 * np.log10(max(snr, 1e-10))
            estimated_psnr = psnr_model(snr_db)
            if estimated_psnr < psnr_req[i]:
                violations += 1
                total_latency += QOS_PENALTY

        avg_latency = total_latency / max(num_nodes, 1)
        # 惩罚函数 (Eq. 42)
        fitness = avg_latency + QOS_PENALTY * violations
        return fitness, avg_latency, violations

    def _fitness_stage2(self, solution, gw_freqs, gw_distances, noise_w,
                        bandwidth, data_length, psnr_req, psnr_model):
        """
        Stage 2 适应度: 网关→低SNR GU 的子载波分配
        """
        num_nodes = len(solution)
        total_latency = 0.0
        violations = 0
        used_subcarriers = set()

        for i in range(num_nodes):
            j = solution[i]
            if j < 0:
                violations += 1
                total_latency += QOS_PENALTY
                continue

            if j in used_subcarriers:
                violations += 1
                total_latency += QOS_PENALTY * 0.5
                continue
            used_subcarriers.add(j)

            # 网关到GU的SNR
            path_loss = (4 * np.pi * gw_distances[i] * gw_freqs[j] / SPEED_OF_LIGHT) ** 2
            G_g = 10 ** (GW_ANTENNA_GAIN_DBI / 10)
            G_u = 10 ** (GU_ANTENNA_GAIN_DBI / 10)
            snr = (GW_TX_POWER * G_g * G_u) / (noise_w[i] * path_loss)

            rate = bandwidth * np.log2(1 + max(snr, 1e-10))
            if rate <= 0:
                violations += 1
                total_latency += QOS_PENALTY
                continue

            latency = data_length / rate
            total_latency += latency

            snr_db = 10 * np.log10(max(snr, 1e-10))
            estimated_psnr = psnr_model(snr_db)
            if estimated_psnr < psnr_req[i]:
                violations += 1
                total_latency += QOS_PENALTY

        avg_latency = total_latency / max(num_nodes, 1)
        fitness = avg_latency + QOS_PENALTY * violations
        return fitness, avg_latency, violations

    def _compute_snr(self, freq, distance, gain_r, noise_w):
        """计算SNR"""
        G_s = 10 ** (SAT_ANTENNA_GAIN_DBI / 10)
        path_loss = (4 * np.pi * distance * freq / SPEED_OF_LIGHT) ** 2
        snr = (SAT_TX_POWER * G_s * gain_r) / (noise_w * path_loss)
        return snr

    def optimize_stage1(self, num_nodes, num_subcarriers, sat_freqs, distances,
                        noise_w, gain_receiver, bandwidth, data_length,
                        psnr_req, psnr_model):
        """
        Stage 1 优化: 卫星到地面节点的子载波分配
        """
        pop = self._init_population(num_nodes, num_subcarriers)

        # 评估初始种群
        fitnesses = np.zeros(self.pop_size)
        latencies = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitnesses[i], latencies[i], _ = self._fitness_stage1(
                pop[i], sat_freqs, distances, noise_w, gain_receiver,
                bandwidth, data_length, psnr_req, psnr_model
            )

        best_idx = np.argmin(fitnesses)
        best_solution = pop[best_idx].copy()
        best_fitness = fitnesses[best_idx]

        # DWOA 迭代
        for tau in range(self.max_iter):
            # 线性递减 a: 2 → 0
            a = 2.0 - tau * (2.0 / self.max_iter)

            for i in range(self.pop_size):
                r1 = self.rng.random()
                r2 = self.rng.random()

                A = 2 * a * r1 - a
                C = 2 * r2
                p = self.rng.random()
                l = self.rng.uniform(-1, 1)

                new_sol = pop[i].copy()

                if p < 0.5:
                    if abs(A) < 1:
                        # 开发阶段: 包围猎物 (Eq. 36, 37, 38)
                        D = np.abs(C * best_solution - pop[i])
                        continuous = best_solution - A * D
                        for j in range(num_nodes):
                            new_sol[j] = self._discretize(continuous[j], num_subcarriers)
                    else:
                        # 探索阶段 (Eq. 40, 41)
                        rand_idx = self.rng.randint(0, self.pop_size)
                        D_rand = np.abs(C * pop[rand_idx] - pop[i])
                        continuous = pop[rand_idx] - A * D_rand
                        for j in range(num_nodes):
                            new_sol[j] = self._discretize(continuous[j], num_subcarriers)
                else:
                    # 螺旋更新 (Eq. 36)
                    D_prime = np.abs(best_solution - pop[i])
                    continuous = D_prime * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_solution
                    for j in range(num_nodes):
                        new_sol[j] = self._discretize(continuous[j], num_subcarriers)

                # 评估新解
                new_fitness, new_latency, _ = self._fitness_stage1(
                    new_sol, sat_freqs, distances, noise_w, gain_receiver,
                    bandwidth, data_length, psnr_req, psnr_model
                )

                if new_fitness <= fitnesses[i]:
                    pop[i] = new_sol
                    fitnesses[i] = new_fitness
                    latencies[i] = new_latency

                    if new_fitness < best_fitness:
                        best_solution = new_sol.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness

    def optimize_stage2(self, num_nodes, num_subcarriers, gw_freqs,
                        gw_distances, noise_w, bandwidth, data_length,
                        psnr_req, psnr_model):
        """Stage 2 优化: 网关到低SNR GU的子载波分配"""
        pop = self._init_population(num_nodes, num_subcarriers)

        fitnesses = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitnesses[i], _, _ = self._fitness_stage2(
                pop[i], gw_freqs, gw_distances, noise_w,
                bandwidth, data_length, psnr_req, psnr_model
            )

        best_idx = np.argmin(fitnesses)
        best_solution = pop[best_idx].copy()
        best_fitness = fitnesses[best_idx]

        for tau in range(self.max_iter):
            a = 2.0 - tau * (2.0 / self.max_iter)

            for i in range(self.pop_size):
                r1 = self.rng.random()
                r2 = self.rng.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                p = self.rng.random()
                l = self.rng.uniform(-1, 1)

                new_sol = pop[i].copy()

                if p < 0.5:
                    if abs(A) < 1:
                        D = np.abs(C * best_solution - pop[i])
                        continuous = best_solution - A * D
                        for j in range(num_nodes):
                            new_sol[j] = self._discretize(continuous[j], num_subcarriers)
                    else:
                        rand_idx = self.rng.randint(0, self.pop_size)
                        D_rand = np.abs(C * pop[rand_idx] - pop[i])
                        continuous = pop[rand_idx] - A * D_rand
                        for j in range(num_nodes):
                            new_sol[j] = self._discretize(continuous[j], num_subcarriers)
                else:
                    D_prime = np.abs(best_solution - pop[i])
                    continuous = D_prime * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_solution
                    for j in range(num_nodes):
                        new_sol[j] = self._discretize(continuous[j], num_subcarriers)

                new_fitness, _, _ = self._fitness_stage2(
                    new_sol, gw_freqs, gw_distances, noise_w,
                    bandwidth, data_length, psnr_req, psnr_model
                )

                if new_fitness <= fitnesses[i]:
                    pop[i] = new_sol
                    fitnesses[i] = new_fitness
                    if new_fitness < best_fitness:
                        best_solution = new_sol.copy()
                        best_fitness = new_fitness

        return best_solution, best_fitness


class GreedyAllocator:
    """贪心算法 - 子载波分配基准"""

    def __init__(self, seed=RANDOM_SEED):
        self.rng = np.random.RandomState(seed)

    def allocate(self, num_nodes, num_subcarriers, snr_matrix, bandwidth,
                 data_length, psnr_req, psnr_model):
        """
        贪心分配: 每个节点选择当前最优的可用子载波
        snr_matrix: (num_nodes, num_subcarriers) 各节点在各子载波上的SNR
        """
        allocation = np.full(num_nodes, -1, dtype=int)
        used_subcarriers = set()

        # 按PSNR需求从高到低排序
        priority_order = np.argsort(-psnr_req)

        for node_idx in priority_order:
            best_sc = -1
            best_metric = -np.inf

            for sc in range(num_subcarriers):
                if sc in used_subcarriers:
                    continue
                snr = snr_matrix[node_idx, sc]
                snr_db = 10 * np.log10(max(snr, 1e-10))
                rate = bandwidth * np.log2(1 + max(snr, 1e-10))
                latency = data_length / max(rate, 1e-10)

                estimated_psnr = psnr_model(snr_db)
                metric = estimated_psnr - latency * 10  # 综合指标

                if metric > best_metric:
                    best_metric = metric
                    best_sc = sc

            if best_sc >= 0:
                allocation[node_idx] = best_sc
                used_subcarriers.add(best_sc)

        return allocation


class PrioritizeAllocator:
    """优先级算法 - 高需求用户优先"""

    def __init__(self, seed=RANDOM_SEED):
        self.rng = np.random.RandomState(seed)

    def allocate(self, num_nodes, num_subcarriers, snr_matrix, bandwidth,
                 data_length, psnr_req, psnr_model):
        """
        优先级分配: PSNR需求最高的用户获得最好的子载波
        """
        allocation = np.full(num_nodes, -1, dtype=int)
        used_subcarriers = set()

        # 按PSNR需求从高到低排序
        priority_order = np.argsort(-psnr_req)

        # 计算每个子载波的平均SNR (排序子载波质量)
        avg_snr_per_sc = np.mean(snr_matrix, axis=0)
        sc_quality_order = np.argsort(-avg_snr_per_sc)

        sc_ptr = 0
        for node_idx in priority_order:
            if sc_ptr < num_subcarriers:
                sc = sc_quality_order[sc_ptr]
                allocation[node_idx] = sc
                used_subcarriers.add(sc)
                sc_ptr += 1

        return allocation
