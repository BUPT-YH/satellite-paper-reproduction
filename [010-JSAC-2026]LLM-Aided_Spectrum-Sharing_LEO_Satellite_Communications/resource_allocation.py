"""
资源分配方案仿真 — Fig. 12 & 13
三种传输方案:
1. 顺序传输 (Sequential): A→B→C→D 全带宽依次传输，用 64QAM (最快)
2. 混合传输 (Hybrid): A,B 用 BPSK; C,D 用 16QAM, 全带宽依次传输
3. LLM 决策传输: 按DeepSeek r1输出同时传输，各业务分配不同子载波和调制方式
"""

import numpy as np
from config import (TOTAL_BW, DATA_VOLUMES, MODULATION_BITS,
                    LLM_SCHEME, LLM_TOTAL_SUB, SUBCARRIER_SPACING)


def gb_to_bits(gb):
    """GB → bits (1 GB = 10^9 bytes = 8×10^9 bits)"""
    return gb * 8e9


class TransmissionScheme:
    """传输方案基类"""
    def __init__(self, name):
        self.name = name
        self.services = ['A', 'B', 'C', 'D']
        self.data_bits = {s: gb_to_bits(DATA_VOLUMES[s]) for s in self.services}

    def compute(self):
        """计算各业务的传输时间、等待时间
        返回: dict with keys: 'wait_time', 'trans_time', 'completion_time'
        """
        raise NotImplementedError


class SequentialScheme(TransmissionScheme):
    """顺序传输方案: A→B→C→D，每种数据用全带宽 64QAM 传输"""

    def __init__(self):
        super().__init__("Sequential (64QAM Full-Band)")
        # 全带宽 64QAM
        self.mod = '64QAM'
        self.rate = MODULATION_BITS[self.mod] * TOTAL_BW  # bits/s

    def compute(self):
        result = {'wait_time': {}, 'trans_time': {}, 'completion_time': {}}
        cumulative_time = 0.0

        for s in self.services:
            t_trans = self.data_bits[s] / self.rate
            result['wait_time'][s] = cumulative_time
            result['trans_time'][s] = t_trans
            cumulative_time += t_trans
            result['completion_time'][s] = cumulative_time

        result['total_time'] = cumulative_time
        return result


class HybridScheme(TransmissionScheme):
    """混合传输方案: A,B 用 BPSK; C,D 用 16QAM, 全带宽依次传输"""

    def __init__(self):
        super().__init__("Hybrid (BPSK+16QAM Full-Band)")
        self.service_mod = {
            'A': 'BPSK',
            'B': 'BPSK',
            'C': '16QAM',
            'D': '16QAM',
        }

    def compute(self):
        result = {'wait_time': {}, 'trans_time': {}, 'completion_time': {}}
        cumulative_time = 0.0

        for s in self.services:
            mod = self.service_mod[s]
            rate = MODULATION_BITS[mod] * TOTAL_BW
            t_trans = self.data_bits[s] / rate
            result['wait_time'][s] = cumulative_time
            result['trans_time'][s] = t_trans
            cumulative_time += t_trans
            result['completion_time'][s] = cumulative_time

        result['total_time'] = cumulative_time
        return result


class LLMScheme(TransmissionScheme):
    """LLM 决策传输方案 (DeepSeek r1)
    所有业务同时传输，各分配不同子载波数和调制方式:
    A: BPSK,   1  subcarrier → BW_A = 1/108 × 2.16 GHz
    B: 16QAM, 30 subcarriers → BW_B = 30/108 × 2.16 GHz
    C: 4QAM,  10 subcarriers → BW_C = 10/108 × 2.16 GHz
    D: 64QAM, 67 subcarriers → BW_D = 67/108 × 2.16 GHz
    """

    def __init__(self):
        super().__init__("LLM Decision (DeepSeek r1)")

    def compute(self):
        result = {'wait_time': {}, 'trans_time': {}, 'completion_time': {}}

        for s in self.services:
            scheme = LLM_SCHEME[s]
            mod = scheme['mod']
            n_sub = scheme['n_sub']
            bw = n_sub / LLM_TOTAL_SUB * TOTAL_BW  # 分配的带宽 (Hz)
            rate = MODULATION_BITS[mod] * bw         # bits/s

            t_trans = self.data_bits[s] / rate
            result['wait_time'][s] = 0.0  # 同时传输，无等待
            result['trans_time'][s] = t_trans
            result['completion_time'][s] = t_trans

        result['total_time'] = max(result['completion_time'].values())
        return result


def compute_transmitted_data_over_time(scheme, t_array):
    """计算给定时间序列上的累计传输数据量 (GB)

    用于绘制 Fig. 12

    参数:
        scheme: TransmissionScheme 实例
        t_array: 时间序列 (s)
    返回:
        cumulative_data (GB) 在每个时间点的值
    """
    result = scheme.compute()
    cumulative = np.zeros_like(t_array)

    if isinstance(scheme, SequentialScheme) or isinstance(scheme, HybridScheme):
        # 顺序传输: 各业务依次传输
        for s in scheme.services:
            mod_name = scheme.mod if isinstance(scheme, SequentialScheme) else scheme.service_mod[s]
            rate = MODULATION_BITS[mod_name] * TOTAL_BW  # bits/s
            t_start = result['wait_time'][s]
            t_end = result['completion_time'][s]

            # 在 [t_start, t_end] 区间内线性增长
            mask = (t_array >= t_start) & (t_array < t_end)
            if np.any(mask):
                fraction = (t_array[mask] - t_start) / (t_end - t_start) if t_end > t_start else 1.0
                cumulative[mask] += DATA_VOLUMES[s] * fraction

            # t >= t_end: 全部完成
            mask_done = t_array >= t_end
            cumulative[mask_done] += DATA_VOLUMES[s]
    else:
        # LLM: 所有业务同时传输
        for s in scheme.services:
            t_trans = result['trans_time'][s]
            rate_gbps = DATA_VOLUMES[s] / t_trans if t_trans > 0 else 0  # GB/s

            mask_active = (t_array >= 0) & (t_array < t_trans)
            cumulative[mask_active] += rate_gbps * t_array[mask_active]

            mask_done = t_array >= t_trans
            cumulative[mask_done] += DATA_VOLUMES[s]

    return cumulative


def compute_delay_summary():
    """计算三种方案的传输时延和等待时延 (用于 Fig. 13)

    返回: dict[scheme_name] -> {service: (wait, trans)}
    """
    schemes = {
        'Sequential': SequentialScheme(),
        'Hybrid': HybridScheme(),
        'LLM Decision': LLMScheme(),
    }
    summary = {}
    for name, scheme in schemes.items():
        result = scheme.compute()
        summary[name] = result
    return summary


# ==================== 测试 ====================

if __name__ == "__main__":
    print("=== 三种传输方案对比 ===\n")

    schemes = [
        ("顺序传输 (64QAM)", SequentialScheme()),
        ("混合传输 (BPSK+16QAM)", HybridScheme()),
        ("LLM 决策 (DeepSeek r1)", LLMScheme()),
    ]

    for name, scheme in schemes:
        result = scheme.compute()
        print(f"--- {name} ---")
        print(f"  总完成时间: {result['total_time']:.2f} s")
        total_delay = sum(result['wait_time'][s] + result['trans_time'][s]
                         for s in scheme.services)
        print(f"  总时延 (Σ 等待+传输): {total_delay:.2f} s")
        for s in scheme.services:
            print(f"  业务 {s}: 等待={result['wait_time'][s]:.4f}s, "
                  f"传输={result['trans_time'][s]:.4f}s, "
                  f"完成={result['completion_time'][s]:.4f}s")
        print()
