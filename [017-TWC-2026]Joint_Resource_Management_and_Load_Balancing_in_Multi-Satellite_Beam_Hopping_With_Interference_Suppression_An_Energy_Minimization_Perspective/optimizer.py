"""
核心优化算法: BCD + Lyapunov 最优功率分配
论文 Section IV: Block Coordinate Descent 算法

统一功率分配框架: 所有方法 (proposed + baselines) 共享 Lyapunov 最优功率计算
差异仅来自 BH 模式选择和负载均衡策略
"""

import numpy as np
import cvxpy as cp
from config import *


def compute_interference(F, P, h, s, k, l, net):
    """计算小区 k 在频段 l 受到的干扰"""
    intra = 0.0
    for j in range(net.K):
        if j != k and F[s, j] > 0.5:
            intra += h[s, k, j] * P[s, j, l]
    inter = 0.0
    for r in net.phi.get(k, []):
        if r != s:
            for j in range(net.K):
                if F[r, j] > 0.5:
                    inter += h[r, k, j] * P[r, j, l]
    return intra + inter


def compute_rate(F, P, h, net):
    """计算可达速率 R_{s,k}"""
    S, K, L = net.S, net.K, net.L
    R = np.zeros((S, K))
    for s in range(S):
        for k in range(K):
            if F[s, k] < 0.5:
                continue
            for l in range(L):
                signal = h[s, k, k] * P[s, k, l]
                interf = compute_interference(F, P, h, s, k, l, net)
                gamma = signal / (NOISE_POWER + interf + 1e-30)
                R[s, k] += BANDWIDTH_PER_SEG * np.log2(1 + gamma)
    return R


def lyapunov_power(F, Q, h, V, net, z_max_lin=None, g=None):
    """统一 Lyapunov 最优功率分配 — 所有方法共享"""
    S, K, L = net.S, net.K, net.L
    P = np.zeros((S, K, L))

    for s in range(S):
        active = [k for k in range(K) if F[s, k] > 0.5]
        if not active:
            continue

        for k in active:
            h_kk = h[s, k, k]
            if h_kk < 1e-30:
                continue
            for l in range(L):
                interf = compute_interference(F, P, h, s, k, l, net)
                denom = V * (NOISE_POWER + interf) * np.log(2) * M0
                if denom > 1e-30 and Q[s, k] > 0:
                    one_plus_gamma = Q[s, k] * BANDWIDTH_PER_SEG * h_kk * T0 / denom
                    gamma = max(one_plus_gamma - 1, 0)
                else:
                    gamma = 0
                P[s, k, l] = max(gamma * (NOISE_POWER + interf) / h_kk, 0)

        # 等比缩放确保总功率不超过 P_MAX
        total = np.sum(P[s])
        if total > P_MAX:
            P[s] *= P_MAX / total

    # 干扰约束: 缩放违反 GSO 干扰阈值的波束功率
    if z_max_lin is not None and g is not None:
        for k_gso in net.k_gso:
            overlap_l = net.l_overlap.get(k_gso, [])
            total_interf = sum(
                g[s, k_gso, j] * P[s, j, l]
                for s in range(S) for j in range(K)
                for l in overlap_l if F[s, j] > 0.5 and l < L
            )
            if total_interf > z_max_lin:
                scale = z_max_lin / total_interf * 0.95
                for l in overlap_l:
                    if l < L:
                        for s in range(S):
                            for j in range(K):
                                if F[s, j] > 0.5:
                                    P[s, j, l] *= scale

    return np.maximum(P, 0)


def qp_load_balancing(F, P, Q, h, net, c_max):
    """QP 负载均衡"""
    S, K = net.S, net.K

    valid_idx = []
    for k in range(K):
        for r in net.phi.get(k, []):
            for s in net.phi.get(k, []):
                if r != s:
                    valid_idx.append((r, s, k))

    n_vars = len(valid_idx)
    if n_vars == 0:
        return np.zeros((S, S, K))

    b = cp.Variable(n_vars)
    b_plus = cp.Variable(n_vars, nonneg=True)

    R = compute_rate(F, P, h, net)

    d_expr = {}
    for s in range(S):
        for k in net.omega[s]:
            d_expr[(s, k)] = cp.Constant(0.0)

    for idx, (r, s, k) in enumerate(valid_idx):
        d_expr[(s, k)] = d_expr.get((s, k), cp.Constant(0.0)) + b[idx]

    obj = 0.0
    for s in range(S):
        for k in net.omega[s]:
            d_e = d_expr.get((s, k), cp.Constant(0.0))
            obj += Q[s, k] * d_e + 0.5 * cp.square(d_e)

    constraints = [b_plus >= b, b_plus >= -b]
    for r in range(S):
        for s2 in range(S):
            if r == s2:
                continue
            terms = [b_plus[idx] for idx, (ri, si, _) in enumerate(valid_idx)
                     if ri == r and si == s2]
            if not terms:
                continue
            c_expr = sum(terms)
            constraints.append(c_expr <= c_max)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=200)
        if b.value is not None:
            B = np.zeros((S, S, K))
            for idx, (r, s, k) in enumerate(valid_idx):
                B[r, s, k] = b.value[idx]
            return B
    except Exception:
        pass
    return np.zeros((S, S, K))


def simple_lb(Q, net, c_max):
    """简单负载均衡 (非 QP)"""
    S, K = net.S, net.K
    B = np.zeros((S, S, K))
    for k in range(K):
        sats = net.phi.get(k, [])
        if len(sats) < 2:
            continue
        avg_q = np.mean([Q[s, k] for s in sats])
        for s in sats:
            diff = Q[s, k] - avg_q
            transfer = np.clip(diff * 0.3, -c_max, c_max)
            for r in sats:
                if r != s:
                    B[r, s, k] = -transfer / max(len(sats) - 1, 1)
    return B


class BCDOptimizer:
    """BCD 优化器: 交替优化 F, P, B — 使用统一功率分配"""

    def __init__(self, network, V=V_DEFAULT, z_max_lin=None, c_max=C_MAX_DEFAULT):
        self.net = network
        self.V = V
        self.z_max = z_max_lin
        self.c_max = c_max
        self.S = network.S
        self.K = network.K
        self.Nb = network.Nb
        self.L = network.L

    def compute_rate(self, F, P, h):
        return compute_rate(F, P, h, self.net)

    def compute_energy(self, P, B):
        S = self.S
        E = np.zeros(S)
        for s in range(S):
            E[s] = np.sum(P[s, :, :]) + P_CIRCUIT
            for r in range(S):
                if r != s and np.sum(np.abs(B[r, s, :])) > 0.5:
                    E[s] += 2 * P_ISL
        return E

    def compute_objective(self, F, P, B, Q, h):
        R = compute_rate(F, P, h, self.net)
        E = self.compute_energy(P, B)
        Lambda = 0.0
        for s in range(self.S):
            for k in self.net.omega[s]:
                if F[s, k] < 0.5:
                    continue
                d_sk = sum(B[r, s, k] for r in self.net.phi.get(k, []) if r != s)
                x_max_rate = R[s, k] * T0 / M0
                x_max_queue = Q[s, k] + d_sk
                x_sk = max(min(x_max_rate, x_max_queue), 0)
                Lambda += Q[s, k] * (d_sk - x_sk) + 0.5 * d_sk ** 2
            Lambda += self.V * E[s]
        return Lambda

    def solve_bh_pattern(self, P_star, B_star, Q, h, g, F_init=None):
        """贪心 + 局部搜索优化 BH 模式"""
        S, K = self.S, self.K
        F = np.zeros((S, K))
        for s in range(S):
            scores = []
            for k in self.net.omega[s]:
                ch_quality = h[s, k, k] if h[s, k, k] > 0 else 1e-30
                score = Q[s, k] * np.log2(1 + ch_quality * P_MAX / (self.Nb * self.L * NOISE_POWER))
                scores.append((k, score))
            scores.sort(key=lambda x: -x[1])
            for k, _ in scores[:self.Nb]:
                F[s, k] = 1.0

        # 局部搜索
        for _ in range(3):
            improved = False
            for s in range(S):
                active = [k for k in range(K) if F[s, k] > 0.5]
                inactive = [k for k in self.net.omega[s] if F[s, k] < 0.5]
                if not inactive:
                    continue
                for k_out in active:
                    P_test = lyapunov_power(F, Q, h, self.V, self.net, self.z_max, g)
                    obj_cur = self._eval_bh(F, P_test, Q, h, s)

                    F[s, k_out] = 0
                    best_k_in, best_obj = None, obj_cur
                    for k_in in inactive:
                        F[s, k_in] = 1
                        P_try = lyapunov_power(F, Q, h, self.V, self.net, self.z_max, g)
                        obj_new = self._eval_bh(F, P_try, Q, h, s)
                        if obj_new < best_obj:
                            best_obj = obj_new
                            best_k_in = k_in
                        F[s, k_in] = 0

                    if best_k_in is not None:
                        F[s, best_k_in] = 1
                        improved = True
                    else:
                        F[s, k_out] = 1
            if not improved:
                break
        return F

    def _eval_bh(self, F, P, Q, h, s):
        obj = 0.0
        R = compute_rate(F, P, h, self.net)
        for k in range(self.K):
            if F[s, k] > 0.5:
                x_sk = max(min(R[s, k] * T0 / M0, Q[s, k]), 0)
                obj += Q[s, k] * (0 - x_sk)
        obj += self.V * np.sum(P[s, :, :])
        return obj

    def solve_resource_mgmt(self, F_star, B_star, Q, h, g, P_init=None):
        """使用统一 Lyapunov 功率分配"""
        return lyapunov_power(F_star, Q, h, self.V, self.net,
                              z_max_lin=self.z_max, g=g)

    def solve_load_balancing(self, F_star, P_star, Q, h):
        return qp_load_balancing(F_star, P_star, Q, h, self.net, self.c_max)

    def solve_bcd(self, Q, h, g, F_init=None, P_init=None, B_init=None):
        """BCD 算法 (Algorithm 1)"""
        S, K, L = self.S, self.K, self.L

        if F_init is None:
            F = np.zeros((S, K))
            for s in range(S):
                cells_q = [(k, Q[s, k]) for k in self.net.omega[s]]
                cells_q.sort(key=lambda x: -x[1])
                for k, _ in cells_q[:self.Nb]:
                    F[s, k] = 1.0
        else:
            F = F_init.copy()

        B = B_init.copy() if B_init is not None else np.zeros((S, S, K))
        obj_history = []

        for i in range(BCD_MAX_ITER):
            # Step 1: 优化 F
            F_new = self.solve_bh_pattern(
                lyapunov_power(F, Q, h, self.V, self.net, self.z_max, g),
                B, Q, h, g, F_init=F)
            P_for_obj = lyapunov_power(F_new, Q, h, self.V, self.net, self.z_max, g)
            obj_history.append(self.compute_objective(F_new, P_for_obj, B, Q, h))

            # Step 2: 优化 P
            P_new = self.solve_resource_mgmt(F_new, B, Q, h, g)
            obj_history.append(self.compute_objective(F_new, P_new, B, Q, h))

            # Step 3: 优化 B
            B_new = self.solve_load_balancing(F_new, P_new, Q, h)
            obj_history.append(self.compute_objective(F_new, P_new, B_new, Q, h))

            F, B = F_new, B_new

            if i > 0 and len(obj_history) >= 6:
                if abs(obj_history[-3] - obj_history[-1]) < CONV_TOL * max(abs(obj_history[-3]), 1):
                    break

        P_final = lyapunov_power(F, Q, h, self.V, self.net, self.z_max, g)
        return F, P_final, B, obj_history
