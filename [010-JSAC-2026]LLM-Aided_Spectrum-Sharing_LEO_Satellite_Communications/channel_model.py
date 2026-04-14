"""
信道模型与卫星位置分布
- Shadowed Rician 衰落信道的 PDF/CDF
- 卫星位置联合分布 fr,θ 的 PDF 及采样
- 干扰区域体积 V 的计算
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammainc, exp1
from scipy.integrate import quad
from config import (RE, RL, RH, THETA0, MN, OMEGA, B_PARAM,
                    BETA, DELTA, ALPHA_H, ZETA)


# ==================== 辅助函数 ====================

def r_func(theta):
    """Eq.(2): r(θ) = RE / (sinΘ0 × sin(θ+Θ0))
    给定角度 θ, 返回轨道半径 r"""
    return RE / (np.sin(THETA0) * np.sin(theta + THETA0))


def theta_func(r):
    """Eq.(2) 逆函数: θ(r) = arcsin(r × sinΘ0 / RE) - Θ0
    给定轨道半径 r, 返回角度 θ"""
    val = r * np.sin(THETA0) / RE
    val = np.clip(val, -1, 1)
    return np.arcsin(val) - THETA0


# ==================== 干扰区域体积 V ====================

def compute_volume():
    """Eq.(4): 计算干扰区域 ΩI 的体积 V
    V = 2π/3·(RH³-RL³) - 2π·∫_{RL}^{RH} r²·cos(θ(r))dr

    第二项用数值积分计算
    """
    term1 = 2 * np.pi / 3 * (RH**3 - RL**3)

    def integrand(r):
        th = theta_func(r)
        return r**2 * np.cos(th)

    integral_val, _ = quad(integrand, RL, RH)
    term2 = 2 * np.pi * integral_val

    V = term1 - term2
    return V

# 全局计算体积
V_VOLUME = compute_volume()


# ==================== 卫星位置 PDF ====================

def satellite_pdf(r, theta):
    """Eq.(19): 卫星位置 (r, θ) 的联合 PDF
    fr,θ(r, θ) = 2π/V × r² × sin(θ)
    有效范围: RL ≤ r < RH, 0 ≤ θ < θ(r)
    """
    theta_r = theta_func(r)
    valid = (r >= RL) & (r < RH) & (theta >= 0) & (theta < theta_r)
    pdf = np.where(valid, 2 * np.pi / V_VOLUME * r**2 * np.sin(theta), 0.0)
    return pdf


# ==================== 卫星位置采样 (蒙特卡洛用) ====================

def sample_satellite_positions(n_samples, rng=None):
    """从 fr,θ 分布中采样卫星位置

    采用逆变换+拒绝采样:
    1. 径向 r: 边际分布 f(r) ∝ r²·(1-cos(θ(r))), 用拒绝采样
    2. 角度 θ|r: CDF = (1-cos(θ))/(1-cos(θ(r))), 直接逆变换

    返回: (r_array, theta_array), 各形状 (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 预计算径向分布 f(r) ∝ r²·(1-cos(θ(r)))
    r_grid = np.linspace(RL, RH, 10000)
    theta_grid = theta_func(r_grid)
    f_r = r_grid**2 * (1 - np.cos(theta_grid))
    f_max = np.max(f_r)

    # 拒绝采样 r
    r_samples = np.zeros(n_samples)
    count = 0
    while count < n_samples:
        batch = max(n_samples - count, 1000)
        r_proposal = rng.uniform(RL, RH, batch)
        theta_proposal = theta_func(r_proposal)
        f_val = r_proposal**2 * (1 - np.cos(theta_proposal))
        u = rng.uniform(0, f_max, batch)
        accepted = r_proposal[u <= f_val]
        n_accept = min(len(accepted), n_samples - count)
        r_samples[count:count + n_accept] = accepted[:n_accept]
        count += n_accept

    # 给定 r, 用逆变换采样 θ
    theta_r = theta_func(r_samples)
    u = rng.uniform(0, 1, n_samples)
    theta_samples = np.arccos(1 - u * (1 - np.cos(theta_r)))

    return r_samples, theta_samples


def compute_satellite_distance(r, theta):
    """由卫星位置 (r, θ) 计算到地面 D 的距离

    D 在地球表面，距地心 RE, 沿 x 轴正方向
    卫星在 (r·cos(θ), r·sin(θ)) 处 (二维截面)

    d² = r² + RE² - 2·r·RE·cos(θ)
    """
    return np.sqrt(r**2 + RE**2 - 2 * r * RE * np.cos(theta))


# ==================== Shadowed Rician 信道 ====================

def shadowed_rician_cdf(x):
    """Eq.(26): |h|² 的 CDF
    F(x) = Σ_{k=0}^{mn-1} ζ(k)/(β-δ)^{k+1} · γ(k+1, (β-δ)·x)
    """
    result = 0.0
    beta_delta = BETA - DELTA
    for k in range(MN):
        # 下不完全伽马函数 γ(k+1, (β-δ)·x) = Γ(k+1) · P(k+1, (β-δ)·x)
        # scipy.special.gammainc(a, x) = γ(a,x)/Γ(a) (正则化下不完全伽马函数)
        a = k + 1
        z = beta_delta * x
        if z <= 0:
            gamma_val = 0.0
        else:
            gamma_val = gamma_func(a) * gammainc(a, z)
        result += ZETA[k] / (beta_delta**(k + 1)) * gamma_val
    return result


def shadowed_rician_cdf_array(x_arr):
    """向量化的 Shadowed Rician CDF"""
    x_arr = np.asarray(x_arr, dtype=float)
    result = np.zeros_like(x_arr)
    beta_delta = BETA - DELTA
    for k in range(MN):
        a = k + 1
        z = beta_delta * x_arr
        gamma_val = gamma_func(a) * gammainc(a, z)
        result += ZETA[k] / (beta_delta**(k + 1)) * gamma_val
    return result


def sample_shadowed_rician(n_samples, rng=None):
    """采样 Shadowed Rician 信道增益 |h|²

    使用逆变换法: 生成均匀分布 U ~ Uniform(0,1)
    然后 |h|² = F^{-1}(U)
    由于 F^{-1} 没有闭式解, 用数值方法
    """
    if rng is None:
        rng = np.random.default_rng()

    from scipy.optimize import brentq

    u = rng.uniform(0, 1, n_samples)
    samples = np.zeros(n_samples)

    for i in range(n_samples):
        if u[i] <= 1e-15:
            samples[i] = 0.0
        elif u[i] >= 1 - 1e-15:
            samples[i] = 100.0  # 上界截断
        else:
            try:
                samples[i] = brentq(lambda x: shadowed_rician_cdf(x) - u[i], 0, 100)
            except ValueError:
                samples[i] = 0.0

    return samples


def sample_shadowed_rician_fast(n_samples, rng=None):
    """快速采样 Shadowed Rician 信道增益 |h|²
    利用 PDF 形式直接生成混合分布样本

    f(x) = Σ ζ(k) · x^k · exp(-(β-δ)·x)
    即混合 Gamma(k+1, 1/(β-δ)) 分布加权和
    """
    if rng is None:
        rng = np.random.default_rng()

    beta_delta = BETA - DELTA
    # ζ(k) 可正可负，不能直接用作混合权重
    # 改用逆变换法，但预计算 CDF 表加速
    n_table = 10000
    x_max = 20.0 / beta_delta  # 覆盖大部分概率质量
    x_table = np.linspace(0, x_max, n_table)
    cdf_table = shadowed_rician_cdf_array(x_table)

    # 插值逆变换
    u = rng.uniform(0, 1, n_samples)
    # 确保单调
    cdf_table = np.maximum.accumulate(cdf_table)
    samples = np.interp(u, cdf_table, x_table)

    return samples


# ==================== 验证函数 ====================

def verify_channel_model():
    """验证信道模型: PDF 积分应接近 1"""
    from scipy.integrate import quad

    def pdf_func(x):
        """Eq.(25): |h|² 的 PDF"""
        result = 0.0
        beta_delta = BETA - DELTA
        for k in range(MN):
            result += ZETA[k] * x**k * np.exp(-beta_delta * x)
        return result

    integral, _ = quad(pdf_func, 0, 200)
    print(f"[验证] |h|² PDF 积分 = {integral:.6f} (应接近 1.0)")
    print(f"[验证] β = {BETA:.4f}, δ = {DELTA:.4f}, β-δ = {BETA-DELTA:.4f}")
    print(f"[验证] αh = {ALPHA_H:.6f}")
    print(f"[验证] V = {V_VOLUME:.4e} m³")
    print(f"[验证] θ(RL) = {np.rad2deg(theta_func(RL)):.4f}°, θ(RH) = {np.rad2deg(theta_func(RH)):.4f}°")


if __name__ == "__main__":
    verify_channel_model()
