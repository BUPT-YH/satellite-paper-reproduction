"""
中断概率分析 (优化版)
- Pout,1: 频谱共享传输中断概率 (Eq.40)
- Pout,2: 固定频段传输中断概率 (Eq.45-46) — 向量化加速
- 蒙特卡洛仿真验证
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc
from config import (RE, RL, RH, THETA0, ALPHA, MN, BETA, DELTA,
                    ZETA, SIGMA_N2, MF, LAMBDA_E_FS, PE_OVER_PS,
                    DSD_DEFAULT)
from channel_model import V_VOLUME, theta_func


# ==================== 预计算 Chebyshev 节点 ====================

_q_idx = np.arange(1, MF + 1)
_chi = np.cos((2 * _q_idx - 1) * np.pi / (2 * MF))
_sqrt_1_chi2 = np.sqrt(1 - _chi**2)
_tau = (RH - RL) / 2 * _chi + (RH + RL) / 2
_tau_theta = theta_func(_tau)


def xi_coeff(k):
    """ξ(k) = [Γ(k+1)]^{-1/(k+1)}"""
    return gamma_func(k + 1) ** (-1.0 / (k + 1))


# ==================== Pout,1: 频谱共享传输 OP ====================

def pout1_analytical(PS_dB, gamma_th_dB, dSD):
    """Eq.(40): 频谱共享传输中断概率"""
    PS = 10 ** (PS_dB / 10)
    gamma_th = 10 ** (gamma_th_dB / 10)
    beta_delta = BETA - DELTA
    x = beta_delta * SIGMA_N2 * dSD**ALPHA * gamma_th / PS

    pout = 0.0
    for k in range(MN):
        a = k + 1
        gamma_val = gamma_func(a) * gammainc(a, x)
        pout += ZETA[k] / (beta_delta ** (k + 1)) * gamma_val
    return pout


def pout1_analytical_array(PS_dB_arr, gamma_th_dB, dSD):
    """向量化的 Pout,1 (PS 为数组)"""
    PS_dB_arr = np.asarray(PS_dB_arr, dtype=float)
    PS = 10 ** (PS_dB_arr / 10)
    gamma_th = 10 ** (gamma_th_dB / 10)
    beta_delta = BETA - DELTA
    x = beta_delta * SIGMA_N2 * dSD**ALPHA * gamma_th / PS

    pout = np.zeros_like(x)
    for k in range(MN):
        a = k + 1
        gamma_val = gamma_func(a) * gammainc(a, x)
        pout += ZETA[k] / (beta_delta ** (k + 1)) * gamma_val
    return pout


# ==================== Pout,2: 固定频段传输 OP (向量化) ====================

def compute_Ei_vectorized(k_val, t_val, gamma_th_lin, PS, dSD, PE_ratio):
    """Eq.(46): 向量化计算 Ei

    使用 NumPy 广播一次性计算所有 q1, q2 组合
    """
    beta_delta = BETA - DELTA
    xi_k = xi_coeff(k_val)
    Xi1 = gamma_th_lin * PE_ratio * dSD**ALPHA

    # q1: 径向, q2: 角度 — 广播 (MF, 1) x (1, MF)
    tau_col = _tau[:, np.newaxis]          # (MF, 1)
    tau_theta_col = _tau_theta[:, np.newaxis]  # (MF, 1)
    sqrt1_col = _sqrt_1_chi2[:, np.newaxis]    # (MF, 1)
    chi_row = _chi[np.newaxis, :]          # (1, MF)
    sqrt1_row = _sqrt_1_chi2[np.newaxis, :]  # (1, MF)

    # angle = θ(τ)/2 · χ + θ(τ)/2
    angle = tau_theta_col / 2 * chi_row + tau_theta_col / 2  # (MF, MF)

    # 卫星到地面距离^α
    d_ei2 = tau_col**2 + RE**2 - 2 * tau_col * RE * np.cos(angle)
    d_ei_alpha = d_ei2 ** (ALPHA / 2)

    # sin(angle)
    sin_term = np.sin(angle)

    # base = [1 + ξ(k)·Ξ1·t/d_ei^α]
    base = 1 + xi_k * Xi1 * t_val / d_ei_alpha  # (MF, MF)

    # 权重: τ²·θ(τ)·√(1-χ²)·√(1-χ²)·sin(angle)
    weight = tau_col**2 * tau_theta_col * sqrt1_col * sqrt1_row * sin_term  # (MF, MF)

    # 对 p 求和
    Ei = 0.0
    for p in range(MN):
        coeff = (np.pi**3 * ZETA[p] * gamma_func(p + 1) * (RH - RL)
                 / (2 * V_VOLUME * beta_delta**(p + 1) * MF**2))
        Ei += coeff * np.sum(weight * base**(-(p + 1)))

    return Ei


def pout2_analytical(gamma_th_dB, PS_dB, dSD, lambda_e_fs=LAMBDA_E_FS,
                     PE_over_PS=PE_OVER_PS):
    """Eq.(45): 固定频段传输中断概率 (解析)"""
    PS = 10 ** (PS_dB / 10)
    gamma_th = 10 ** (gamma_th_dB / 10)
    beta_delta = BETA - DELTA
    Xi2 = gamma_th * SIGMA_N2 * dSD**ALPHA / PS

    pout = 0.0
    for k in range(MN):
        xi_k = xi_coeff(k)
        outer_coeff = ZETA[k] * gamma_func(k + 1) / beta_delta**(k + 1)

        inner_sum = 0.0
        for t in range(k + 2):
            binom_coeff = gamma_func(k + 2) / (gamma_func(t + 1) * gamma_func(k + 2 - t))
            sign = (-1) ** t
            exp_Xi2 = np.exp(-xi_k * beta_delta * Xi2 * t)
            Ei = compute_Ei_vectorized(k, t, gamma_th, PS, dSD, PE_over_PS)
            exp_Ei = np.exp(lambda_e_fs * (Ei - 1))
            inner_sum += binom_coeff * sign * exp_Xi2 * exp_Ei

        pout += outer_coeff * inner_sum

    return min(pout, 1.0)


# ==================== 蒙特卡洛仿真 ====================

def pout1_montecarlo(PS_dB, gamma_th_dB, dSD, n_trials=200000, rng=None):
    """Pout,1 蒙特卡洛"""
    if rng is None:
        rng = np.random.default_rng()
    PS = 10 ** (PS_dB / 10)
    gamma_th = 10 ** (gamma_th_dB / 10)
    from channel_model import sample_shadowed_rician_fast
    hs2 = sample_shadowed_rician_fast(n_trials, rng)
    SNR = PS * hs2 * dSD**(-ALPHA) / SIGMA_N2
    return np.mean(SNR < gamma_th)


def pout2_montecarlo_batch(gamma_th_dB_arr, PS_dB, dSD,
                           lambda_e_fs=LAMBDA_E_FS,
                           PE_over_PS=PE_OVER_PS,
                           n_trials=50000, rng=None):
    """Pout,2 批量蒙特卡洛"""
    if rng is None:
        rng = np.random.default_rng()
    PS = 10 ** (PS_dB / 10)
    PE = PE_over_PS * PS
    gamma_th_arr = 10 ** (np.asarray(gamma_th_dB_arr, dtype=float) / 10)

    from channel_model import (sample_satellite_positions,
                               compute_satellite_distance,
                               sample_shadowed_rician_fast)

    outage_counts = np.zeros(len(gamma_th_arr))
    for _ in range(n_trials):
        hs2 = sample_shadowed_rician_fast(1, rng)[0]
        N_env = rng.poisson(lambda_e_fs)
        interference = 0.0
        if N_env > 0:
            r_int, theta_int = sample_satellite_positions(N_env, rng)
            d_int = compute_satellite_distance(r_int, theta_int)
            he2 = sample_shadowed_rician_fast(N_env, rng)
            interference = np.sum(PE * he2 * d_int**(-ALPHA))
        SINR = PS * hs2 * dSD**(-ALPHA) / (interference + SIGMA_N2)
        outage_counts += (SINR < gamma_th_arr).astype(int)

    return outage_counts / n_trials


# ==================== 测试 ====================

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("=== Pout,1 Test ===")
    dSD = 800e3
    for ps in [10, 20, 30]:
        for gth in [0, 5, 10]:
            val = pout1_analytical(ps, gth, dSD)
            print(f"PS={ps}dBW, gth={gth}dB: Pout,1={val:.6e}")

    print("\n=== Pout,2 Speed Test ===")
    import time
    t0 = time.time()
    val = pout2_analytical(10, 20, dSD)
    t1 = time.time()
    print(f"Pout,2(10dB): {val:.6e} ({t1-t0:.3f}s)")
