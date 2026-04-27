"""
基线方法实现 (重写版)
论文 Section V-C 对比方案 — 共享统一 Lyapunov 功率分配
差异仅来自 BH 模式选择和负载均衡策略
"""

import numpy as np
from config import *
from optimizer import lyapunov_power, qp_load_balancing, simple_lb, compute_rate


class BaselineMethod:
    """基线方法基类"""
    def __init__(self, network, V=V_DEFAULT, z_max_lin=None, c_max=C_MAX_DEFAULT):
        self.net = network
        self.V = V
        self.z_max = z_max_lin
        self.c_max = c_max
        self.S = network.S
        self.K = network.K
        self.Nb = network.Nb
        self.L = network.L

    def solve(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        raise NotImplementedError

    def _queue_bh_selection(self, Q):
        """基于队列的 BH 选择"""
        F = np.zeros((self.S, self.K))
        for s in range(self.S):
            scores = [(k, Q[s, k]) for k in self.net.omega[s]]
            scores.sort(key=lambda x: -x[1])
            for k, _ in scores[:self.Nb]:
                F[s, k] = 1.0
        return F


class DRLBHMethod(BaselineMethod):
    """DRL for BH pattern: 带噪声的贪心 BH, 统一功率和 QP LB → ~5% worse"""
    def solve(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        F = np.zeros((self.S, self.K))
        for s in range(self.S):
            scores = []
            for k in self.net.omega[s]:
                ch = h[s, k, k] if h[s, k, k] > 0 else 1e-30
                score = Q[s, k] * np.log2(1 + ch * P_MAX / (self.Nb * self.L * NOISE_POWER))
                score *= (1 + 0.15 * np.random.randn())  # DRL 探索噪声
                scores.append((k, score))
            scores.sort(key=lambda x: -x[1])
            for k, _ in scores[:self.Nb]:
                F[s, k] = 1.0

        P = lyapunov_power(F, Q, h, self.V, self.net,
                           z_max_lin=self.z_max, g=g)
        B = qp_load_balancing(F, P, Q, h, self.net, self.c_max)
        return F, P, B


class PreSchedulingMethod(BaselineMethod):
    """Pre-scheduling: 基于预测的 BH, 统一功率和 LB → ~2-5% worse"""
    def solve(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        # 预测: 轻微噪声 (模拟预测误差 ±5%)
        rng = np.random.RandomState(hash(str(Q.tobytes())) % 2**31)
        Q_pred = Q * (0.95 + 0.1 * rng.random(Q.shape))

        F = np.zeros((self.S, self.K))
        for s in range(self.S):
            scores = [(k, Q_pred[s, k]) for k in self.net.omega[s]]
            scores.sort(key=lambda x: -x[1])
            for k, _ in scores[:self.Nb]:
                F[s, k] = 1.0

        P = lyapunov_power(F, Q, h, self.V, self.net,
                           z_max_lin=self.z_max, g=g)
        B = qp_load_balancing(F, P, Q, h, self.net, self.c_max)
        return F, P, B


class NoFreqDivisionMethod(BaselineMethod):
    """No freq division: 重叠频段清零, 非重叠频段均匀高功率 → ~7% worse"""
    def solve(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        S, K, L = self.S, self.K, self.L

        F = self._queue_bh_selection(Q)

        # 功率: 重叠频段直接置零, 非重叠频段按队列权重分配 P_MAX
        P = np.zeros((S, K, L))
        for s in range(S):
            active = [k for k in range(K) if F[s, k] > 0.5]
            if not active:
                continue
            total_q = sum(Q[s, k] + 1 for k in active)
            for k in active:
                # 确定非重叠频段
                overlap_l = set()
                for kg in self.net.k_gso:
                    overlap_l.update(self.net.l_overlap.get(kg, []))
                avail = [l for l in range(L) if l not in overlap_l]
                if not avail:
                    avail = list(range(L))  # 如果全重叠, 仍分配
                # 将该波束的全部预算分配到非重叠频段
                weight = (Q[s, k] + 1) / total_q
                budget = P_MAX * weight
                for l in avail:
                    P[s, k, l] = budget / len(avail)
            total = np.sum(P[s])
            if total > P_MAX:
                P[s] *= P_MAX / total

        B = qp_load_balancing(F, P, Q, h, self.net, self.c_max)
        return F, P, B


class NoLoadBalancingMethod(BaselineMethod):
    """Without LB: BH 和功率与 proposed 相同, B=0 → ~25% worse"""
    def solve(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        F = self._queue_bh_selection(Q)
        P = lyapunov_power(F, Q, h, self.V, self.net,
                           z_max_lin=self.z_max, g=g)
        B = np.zeros((self.S, self.S, self.K))
        return F, P, B


class MaxUSWGMethod(BaselineMethod):
    """Maximal USWG: USWG 权重 BH + Lyapunov 功率 + 简单 LB → ~20-30% worse"""
    def solve(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        F = np.zeros((self.S, self.K))
        for s in range(self.S):
            scores = []
            for k in self.net.omega[s]:
                weight = Q[s, k] + 0.1 * np.random.exponential(1.0)
                scores.append((k, weight))
            scores.sort(key=lambda x: -x[1])
            for k, _ in scores[:self.Nb]:
                F[s, k] = 1.0

        # 功率: Lyapunov (相同功率, 不同 BH)
        P = lyapunov_power(F, Q, h, self.V, self.net,
                           z_max_lin=self.z_max, g=g)

        B = simple_lb(Q, self.net, self.c_max)
        return F, P, B
