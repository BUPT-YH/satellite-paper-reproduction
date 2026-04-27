"""
仿真主循环: 时隙级仿真框架
论文 Section V 仿真设置
"""

import numpy as np
from config import *
from channel_model import SatelliteNetwork
from optimizer import BCDOptimizer
from baselines import (DRLBHMethod, PreSchedulingMethod, NoFreqDivisionMethod,
                       NoLoadBalancingMethod, MaxUSWGMethod)


def run_simulation(network, method_name='proposed', V=V_DEFAULT,
                   demand=DEMAND_DEFAULT, z_max_lin=None, c_max=C_MAX_DEFAULT,
                   num_slots=NUM_TIME_SLOTS, verbose=True):
    """
    运行完整的时隙仿真

    Args:
        network: SatelliteNetwork 实例
        method_name: 方法名称
        V: 权衡系数
        demand: 总通信需求 (bps)
        z_max_lin: 干扰阈值 (线性值)
        c_max: ISL 传输上限
        num_slots: 仿真时隙数
        verbose: 是否打印进度

    Returns:
        dict: 仿真结果 (平均功率, 平均队列长度, 每时隙数据等)
    """
    S, K, L = network.S, network.K, network.L

    # 创建优化器
    if method_name == 'proposed':
        method = BCDOptimizer(network, V=V, z_max_lin=z_max_lin, c_max=c_max)
    elif method_name == 'drl':
        method = DRLBHMethod(network, V=V, z_max_lin=z_max_lin, c_max=c_max)
    elif method_name == 'pre_scheduling':
        method = PreSchedulingMethod(network, V=V, z_max_lin=z_max_lin, c_max=c_max)
    elif method_name == 'no_freq_div':
        method = NoFreqDivisionMethod(network, V=V, z_max_lin=z_max_lin, c_max=c_max)
    elif method_name == 'no_lb':
        method = NoLoadBalancingMethod(network, V=V, z_max_lin=z_max_lin, c_max=c_max)
    elif method_name == 'max_uswg':
        method = MaxUSWGMethod(network, V=V, z_max_lin=z_max_lin, c_max=c_max)
    else:
        raise ValueError(f"未知方法: {method_name}")

    # 初始化队列
    Q = np.zeros((S, K))

    # 记录变量
    power_per_slot = np.zeros(num_slots)
    queue_per_slot = np.zeros(num_slots)
    throughput_per_slot = np.zeros(num_slots)

    # 热启动变量
    F_prev, P_prev, B_prev = None, None, None

    for n in range(num_slots):
        # 生成信道系数
        h, g = network.generate_channel_coefficients(time_slot=n)

        # 生成分组到达
        a, rates = network.generate_arrivals(demand_total=demand, time_slot=n)

        # 求解优化
        if method_name == 'proposed':
            F, P, B, _ = method.solve_bcd(Q, h, g,
                                          F_init=F_prev, P_init=P_prev, B_init=B_prev)
        else:
            F, P, B = method.solve(Q, h, g,
                                   F_init=F_prev, P_init=P_prev, B_init=B_prev)

        F_prev, P_prev, B_prev = F.copy(), P.copy(), B.copy()

        # 计算速率
        R = method.compute_rate(F, P, h) if hasattr(method, 'compute_rate') else \
            BCDOptimizer(network).compute_rate(F, P, h)

        # 计算负载变化
        d = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                for r in network.phi.get(k, []):
                    d[s, k] += B[r, s, k]

        # 计算传输包数
        x = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                x_max_rate = R[s, k] * T0 / M0
                x_max_queue = Q[s, k] + d[s, k]
                x[s, k] = max(min(x_max_rate, x_max_queue), 0)

        # 计算能耗
        E = np.zeros(S)
        for s in range(S):
            tx_power = np.sum(P[s, :, :])
            isl_count = 0
            for r in range(S):
                if r != s:
                    c_rs = np.sum(np.abs(B[r, s, :]))
                    if c_rs > 0.5:
                        isl_count += 1
            E[s] = tx_power + P_CIRCUIT + 2 * P_ISL * isl_count

        # 更新队列
        for s in range(S):
            for k in range(K):
                Q[s, k] = max(0, Q[s, k] + a[s, k] + d[s, k] - x[s, k])

        # 记录指标
        power_per_slot[n] = np.mean(E)
        queue_per_slot[n] = np.mean(Q)
        throughput_per_slot[n] = np.mean(x) * M0 / T0  # bps

        if verbose and (n + 1) % 50 == 0:
            print(f"  时隙 {n+1}/{num_slots}: "
                  f"功率={power_per_slot[n]:.1f}W, "
                  f"队列={queue_per_slot[n]:.1f}")

    # 计算稳态平均值 (去掉预热期)
    steady_power = np.mean(power_per_slot[WARMUP_SLOTS:])
    steady_queue = np.mean(queue_per_slot[WARMUP_SLOTS:])
    steady_throughput = np.mean(throughput_per_slot[WARMUP_SLOTS:])

    return {
        'power_per_slot': power_per_slot,
        'queue_per_slot': queue_per_slot,
        'throughput_per_slot': throughput_per_slot,
        'avg_power': steady_power,
        'avg_queue': steady_queue,
        'avg_throughput': steady_throughput,
        'method': method_name,
        'V': V,
        'demand': demand,
    }


def run_bcd_convergence(num_sats_list=None, V=200):
    """
    Fig. 2: BCD 算法收敛曲线
    不同卫星数量下的迭代收敛
    """
    if num_sats_list is None:
        num_sats_list = [2, 3, 4]

    results = {}

    for num_sats in num_sats_list:
        print(f"\n--- BCD 收敛性: {num_sats} 颗卫星 ---")
        net = SatelliteNetwork(num_sats=num_sats)

        # 初始化队列: 0 ~ 4000 均匀分布 (论文描述)
        np.random.seed(42)
        Q = np.random.uniform(0, 4000, (net.S, net.K))

        h, g = net.generate_channel_coefficients(time_slot=0)

        opt = BCDOptimizer(net, V=V)
        _, _, _, obj_history = opt.solve_bcd(Q, h, g)

        results[num_sats] = obj_history
        print(f"  收敛: {len(obj_history)} 步, "
              f"最终 Λ = {obj_history[-1]:.2f}")

    return results
