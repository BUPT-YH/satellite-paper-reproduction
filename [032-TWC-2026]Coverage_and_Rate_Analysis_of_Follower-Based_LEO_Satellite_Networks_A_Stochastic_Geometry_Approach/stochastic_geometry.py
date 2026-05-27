"""
核心随机几何算法模块
Coverage and Rate Analysis of Follower-Based LEO Satellite Networks

包含:
- 接触角PDF (Lemma 1, 3)
- 距离计算
- Shadowed-Rician衰落CDF/PDF (Gamma近似)
- 中断概率 (Theorem 1, 2, Corollary 1)
- 平均速率 (Theorem 3, 4, Corollary 2)
"""

import numpy as np
from scipy import integrate
from scipy.special import gammainc, gamma as gamma_func
from scipy.stats import gamma as gamma_dist
import config as cfg


# ============================================================
# 几何辅助函数
# ============================================================

def distance_from_theta(theta):
    """
    根据中心角计算卫星到地面用户的距离 (式5)
    r(θ) = sqrt(R_sat^2 + R_earth^2 - 2*R_sat*R_earth*cos(θ))

    参数:
        theta: 中心角 (弧度)
    返回:
        距离 (km)
    """
    return np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(theta))


def distance_FU(theta, psi):
    """
    计算Follower到地面用户的距离
    使用近似: r_FU(θ,ψ) = sqrt(R_sat^2 + R_earth^2 - 2*R_sat*R_earth*cos(θ-ψ))
    即用最大距离对应的接触角近似

    参数:
        theta: leader-user接触角 (弧度)
        psi: follower相对leader的球冠偏角 (弧度)
    返回:
        距离 (km)
    """
    return np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2 * cfg.R_sat * cfg.R_earth * np.cos(theta - psi))


def distance_FU_with_phi(theta, psi, phi):
    """
    计算Follower到用户的精确距离 (考虑方位角φ)
    使用球面三角: cos(angular_sep) = cos(θ)*cos(ψ) + sin(θ)*sin(ψ)*cos(φ)

    参数:
        theta: leader-user接触角 (弧度)
        psi: follower相对leader的球冠偏角 (弧度)
        phi: 方位角 (弧度, 0~2π)
    返回:
        距离 (km)
    """
    cos_sep = np.cos(theta) * np.cos(psi) + np.sin(theta) * np.sin(psi) * np.cos(phi)
    cos_sep = np.clip(cos_sep, -1, 1)
    angular_sep = np.arccos(cos_sep)
    return distance_from_theta(angular_sep)


def distance_LF(psi):
    """
    计算Leader到Follower的星间距离
    两个卫星都在轨道高度R_sat处, 球面上角距离为psi
    d = 2 * R_sat * sin(psi/2)

    注意: 这里不是distance_from_theta(psi), 那个是卫星到地面的距离!

    参数:
        psi: Follower相对Leader的球冠偏角 (弧度)
    返回:
        距离 (km)
    """
    return 2 * cfg.R_sat * np.sin(psi / 2)


# ============================================================
# 接触角PDF
# ============================================================

def pdf_theta_LU(theta):
    """
    Leader-user接触角PDF (Lemma 1, 式11)
    f_θ(θ) = (N_L * sin(θ) / 2) * ((1+cos(θ))/2)^(N_L-1)

    这是N_L个球面均匀分布卫星中最近一个的接触角分布
    """
    if theta <= 0 or theta >= np.pi:
        return 0.0
    return (cfg.N_L * np.sin(theta) / 2.0) * ((1.0 + np.cos(theta)) / 2.0) ** (cfg.N_L - 1)


def pdf_theta_min(theta):
    """
    Follower-user最小接触角PDF (Lemma 3, 式14)
    f_θ_min(θ) = (N_L * sin(θ+θ_cap) / 2) * ((1+cos(θ+θ_cap))/2)^(N_L-1)
    θ ∈ (0, θ_max - θ_cap]

    对应上界（距离最近）
    """
    theta_shifted = theta + cfg.theta_cap
    if theta <= 0 or theta > cfg.theta_max - cfg.theta_cap:
        return 0.0
    return (cfg.N_L * np.sin(theta_shifted) / 2.0) * ((1.0 + np.cos(theta_shifted)) / 2.0) ** (cfg.N_L - 1)


def pdf_theta_max_contact(theta):
    """
    Follower-user最大接触角PDF (Lemma 3, 式15)
    f_θ_max(θ) = (N_L * sin(θ-θ_cap) / 2) * ((1+cos(θ-θ_cap))/2)^(N_L-1)
    θ ∈ [θ_cap, θ_max + θ_cap)

    对应下界（距离最远）
    """
    theta_shifted = theta - cfg.theta_cap
    if theta < cfg.theta_cap or theta >= cfg.theta_max + cfg.theta_cap:
        return 0.0
    return (cfg.N_L * np.sin(theta_shifted) / 2.0) * ((1.0 + np.cos(theta_shifted)) / 2.0) ** (cfg.N_L - 1)


def pdf_psi(psi):
    """
    Follower在球冠内的角位置PDF (均匀分布)
    f_ψ(ψ) = sin(ψ) / (1 - cos(θ_cap)), ψ ∈ [0, θ_cap]
    """
    if psi < 0 or psi > cfg.theta_cap:
        return 0.0
    return np.sin(psi) / (1.0 - np.cos(cfg.theta_cap))


# ============================================================
# Shadowed-Rician衰落 (Gamma近似)
# ============================================================

def cdf_W(w):
    """
    Shadowed-Rician衰落的Gamma近似CDF (式8-10)
    F_W(w) = gammainc(m1, w/m2) / Γ(m1) 即正则化不完全Gamma函数

    参数:
        w: 信道功率增益
    返回:
        CDF值 F_W(w)
    """
    if w <= 0:
        return 0.0
    # scipy的gammainc是正则化的: gammainc(a, x) = γ(a,x)/Γ(a)
    return gammainc(cfg.m1_gamma, w / cfg.m2_gamma)


def pdf_W(w):
    """
    Shadowed-Rician衰落的Gamma近似PDF (式8)
    f_W(w) = w^(m1-1) * exp(-w/m2) / (m2^m1 * Γ(m1))

    参数:
        w: 信道功率增益
    返回:
        PDF值 f_W(w)
    """
    if w <= 0:
        return 0.0
    m1 = cfg.m1_gamma
    m2 = cfg.m2_gamma
    return w**(m1 - 1) * np.exp(-w / m2) / (m2**m1 * gamma_func(m1))


def sample_W(size=1):
    """
    从Gamma近似中采样信道功率增益

    参数:
        size: 采样数量
    返回:
        采样值数组
    """
    return gamma_dist.rvs(a=cfg.m1_gamma, scale=cfg.m2_gamma, size=size)


# ============================================================
# 中断概率计算
# ============================================================

def outage_leader(gamma_th_dB=None):
    """
    Leader中断概率 (Theorem 1, 式16-17)
    P_out_LU = ∫_0^{θ_max} F_W(γ_th * r_LU(θ)^2 / ξ_LU) * f_θ_LU(θ) dθ

    参数:
        gamma_th_dB: SNR阈值 (dB), 默认使用配置值
    返回:
        中断概率
    """
    if gamma_th_dB is None:
        gamma_th_dB = cfg.gamma_th_dB

    gamma_th = 10**(gamma_th_dB / 10)  # dB -> 线性

    def integrand(theta):
        r = distance_from_theta(theta)
        w_threshold = gamma_th * r**2 / cfg.xi_LU
        return cdf_W(w_threshold) * pdf_theta_LU(theta)

    result, _ = integrate.quad(integrand, 0, cfg.theta_max, limit=200)
    return result


def outage_leader_with_pdf(gamma_th_dB, pdf_func, theta_upper):
    """
    通用的Leader中断概率计算，可使用不同的PDF
    用于计算上下界

    参数:
        gamma_th_dB: SNR阈值 (dB)
        pdf_func: 接触角PDF函数
        theta_upper: 积分上限
    返回:
        中断概率
    """
    gamma_th = 10**(gamma_th_dB / 10)

    def integrand(theta):
        r = distance_from_theta(theta)
        w_threshold = gamma_th * r**2 / cfg.xi_LU
        return cdf_W(w_threshold) * pdf_func(theta)

    result, _ = integrate.quad(integrand, 0, theta_upper, limit=200)
    return result


def outage_cluster(gamma_th_dB=None, N_F=None):
    """
    Cluster中断概率 (Theorem 2, 式18-21)
    P_out_Cluster = ∫_0^{θ_max} (P_out_i_FU(θ))^{N_F} * P_Cond_LU(θ) * f_θ_LU(θ) dθ

    其中:
    P_Cond_LU(θ) = F_W(γ_th * r_LU(θ)^2 / ξ_LU)
    P_out_i_FU(θ) = ∫_0^{θ_cap} F_W(γ_th * r_FU(θ,ψ)^2 / ξ_FU) * f_ψ(ψ) dψ

    参数:
        gamma_th_dB: SNR阈值 (dB)
        N_F: Follower数量
    返回:
        Cluster中断概率
    """
    if gamma_th_dB is None:
        gamma_th_dB = cfg.gamma_th_dB
    if N_F is None:
        N_F = cfg.N_F

    gamma_th = 10**(gamma_th_dB / 10)

    def p_out_i_fu(theta):
        """单个Follower的中断概率 (内层积分)"""
        def inner_integrand(psi):
            r = distance_FU(theta, psi)
            w_threshold = gamma_th * r**2 / cfg.xi_FU
            return cdf_W(w_threshold) * pdf_psi(psi)

        result, _ = integrate.quad(inner_integrand, 0, cfg.theta_cap, limit=100)
        return result

    def integrand(theta):
        p_fu = p_out_i_fu(theta)
        r = distance_from_theta(theta)
        w_threshold = gamma_th * r**2 / cfg.xi_LU
        p_lu = cdf_W(w_threshold)
        return (p_fu ** N_F) * p_lu * pdf_theta_LU(theta)

    result, _ = integrate.quad(integrand, 0, cfg.theta_max, limit=200)
    return result


def outage_upper_bound(gamma_th_dB=None):
    """
    中断概率上界 (Corollary 1, 式22)
    使用θ_min的PDF (最近距离) 代入Leader中断公式
    """
    if gamma_th_dB is None:
        gamma_th_dB = cfg.gamma_th_dB

    theta_upper = cfg.theta_max - cfg.theta_cap
    return outage_leader_with_pdf(gamma_th_dB, pdf_theta_min, theta_upper)


def outage_lower_bound(gamma_th_dB=None):
    """
    中断概率下界 (Corollary 1, 式24)
    使用θ_max的PDF (最远距离) 代入Leader中断公式
    """
    if gamma_th_dB is None:
        gamma_th_dB = cfg.gamma_th_dB

    theta_lower = cfg.theta_cap
    theta_upper = cfg.theta_max + cfg.theta_cap
    gamma_th = 10**(gamma_th_dB / 10)

    def integrand(theta):
        r = distance_from_theta(theta)
        w_threshold = gamma_th * r**2 / cfg.xi_LU
        return cdf_W(w_threshold) * pdf_theta_max_contact(theta)

    result, _ = integrate.quad(integrand, theta_lower, theta_upper, limit=200)
    return result


# ============================================================
# 平均速率计算 (使用快速数值积分)
# ============================================================

# Gamma分布的99.99%分位数作为w积分上限
_W_UPPER = gamma_dist.ppf(0.9999, a=cfg.m1_gamma, scale=cfg.m2_gamma)

# 高斯积分节点数
_N_W = 40      # w方向
_N_THETA = 60  # theta方向
_N_PSI = 20    # psi方向
_N_PHI = 12    # phi方向


def _avg_rate_leader_fast(xi_val, pdf_func, theta_lower=0, theta_upper=None):
    """
    使用向量化的快速数值积分计算Leader平均速率

    参数:
        xi_val: ξ参数值
        pdf_func: 接触角PDF函数
        theta_lower: theta积分下限
        theta_upper: theta积分上限
    返回:
        平均速率 (bps)
    """
    if theta_upper is None:
        theta_upper = cfg.theta_max

    # theta网格
    theta_nodes = np.linspace(theta_lower + 1e-10, theta_upper - 1e-10, _N_THETA)
    dtheta = theta_nodes[1] - theta_nodes[0]

    # w网格 (Gamma分布范围内)
    w_nodes = np.linspace(1e-6, _W_UPPER, _N_W)
    dw = w_nodes[1] - w_nodes[0]

    # 构建网格
    W_grid, T_grid = np.meshgrid(w_nodes, theta_nodes)

    # 计算距离
    R_grid = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2*cfg.R_sat*cfg.R_earth*np.cos(T_grid))

    # 计算SNR
    SNR_grid = xi_val * W_grid / R_grid**2

    # 计算速率
    rate_grid = np.log2(1.0 + SNR_grid)

    # PDF值
    fw_grid = np.array([[pdf_W(w) for w in w_row] for w_row in W_grid])
    ft_vals = np.array([pdf_func(t) for t in theta_nodes])
    ft_grid = np.tile(ft_vals.reshape(-1, 1), (1, _N_W))

    # 被积函数
    integrand_grid = fw_grid * ft_grid * rate_grid

    # 二重积分 (梯形法则)
    inner = np.trapezoid(integrand_grid, w_nodes, axis=1)  # 先对w积分
    result = np.trapezoid(inner, theta_nodes)  # 再对theta积分

    return cfg.B_LU * result


def avg_rate_leader():
    """
    Leader平均速率 (Theorem 3, 式27) - 快速版本
    """
    return _avg_rate_leader_fast(cfg.xi_LU, pdf_theta_LU)


def avg_rate_leader_with_pdf(pdf_func, theta_lower=0, theta_upper=None):
    """
    使用指定PDF计算Leader平均速率 (用于上下界)
    """
    return _avg_rate_leader_fast(cfg.xi_LU, pdf_func, theta_lower, theta_upper)


def _follower_rate_contribution(xi_FU_val, pdf_func, theta_lower=0, theta_upper=None):
    """
    计算单个Follower的速率贡献 (使用网格积分)

    E = integral over theta, psi, phi, w of
        [f_theta(theta) * f_W(w) * sin(psi) / (2*pi*(1-cos(theta_cap)))]
        * min(R_LF, R_FU) dphi dpsi dtheta dw

    参数:
        xi_FU_val: ξ_FU值
        pdf_func: theta的PDF函数
        theta_lower, theta_upper: theta积分限
    返回:
        单个Follower的平均速率贡献 (bps)
    """
    if theta_upper is None:
        theta_upper = cfg.theta_max

    # 网格
    theta_nodes = np.linspace(theta_lower + 1e-10, theta_upper - 1e-10, _N_THETA)
    psi_nodes = np.linspace(1e-6, cfg.theta_cap - 1e-6, _N_PSI)
    phi_nodes = np.linspace(0, 2*np.pi, _N_PHI, endpoint=False)
    w_nodes = np.linspace(1e-6, _W_UPPER, _N_W)

    dphi = 2 * np.pi / _N_PHI

    total = 0.0

    for i_theta, theta in enumerate(theta_nodes):
        ft = pdf_func(theta)
        if ft < 1e-30:
            continue

        r_theta = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2*cfg.R_sat*cfg.R_earth*np.cos(theta))

        for i_psi, psi in enumerate(psi_nodes):
            sin_psi = np.sin(psi)
            if sin_psi < 1e-20:
                continue

            r_lf = 2 * cfg.R_sat * np.sin(psi / 2)  # LF distance on sphere

            # 对phi求平均
            phi_sum = 0.0
            for phi in phi_nodes:
                cos_sep = np.cos(theta)*np.cos(psi) + np.sin(theta)*np.sin(psi)*np.cos(phi)
                cos_sep = np.clip(cos_sep, -1, 1)
                r_fu = np.sqrt(cfg.R_sat**2 + cfg.R_earth**2 - 2*cfg.R_sat*cfg.R_earth*cos_sep)

                # 对w积分
                fw_vals = np.array([pdf_W(w) for w in w_nodes])
                snr_lf = cfg.xi_LF * w_nodes / r_lf**2
                snr_fu = xi_FU_val * w_nodes / r_fu**2

                rate_lf = cfg.B_LF * np.log2(1.0 + snr_lf)
                rate_fu = cfg.B_FU * np.log2(1.0 + snr_fu)
                rate_min = np.minimum(rate_lf, rate_fu)

                w_integral = np.trapezoid(fw_vals * rate_min, w_nodes)
                phi_sum += w_integral

            avg_phi = phi_sum / _N_PHI

            weight = sin_psi / (2 * np.pi * (1 - np.cos(cfg.theta_cap)))
            total += ft * weight * avg_phi * (theta_nodes[1]-theta_nodes[0]) * (psi_nodes[1]-psi_nodes[0])

    return total


def avg_rate_cluster_bounds(N_F=None):
    """
    Cluster平均速率的上下界和中值近似 (Corollary 2)
    使用theta_min(上界), theta_max(下界)计算

    返回:
        (R_upper, R_lower, R_middle) in bps
    """
    if N_F is None:
        N_F = cfg.N_F

    if N_F == 0:
        r = avg_rate_leader()
        return r, r, r

    # Leader速率上下界
    R_upper_LU = avg_rate_leader_with_pdf(
        pdf_theta_min,
        theta_upper=cfg.theta_max - cfg.theta_cap
    )
    R_lower_LU = avg_rate_leader_with_pdf(
        pdf_theta_max_contact,
        theta_lower=cfg.theta_cap,
        theta_upper=cfg.theta_max + cfg.theta_cap
    )

    # Follower贡献上下界
    E_upper = _follower_rate_contribution(
        cfg.xi_FU, pdf_theta_min,
        theta_upper=cfg.theta_max - cfg.theta_cap
    )
    E_lower = _follower_rate_contribution(
        cfg.xi_FU, pdf_theta_max_contact,
        theta_lower=cfg.theta_cap,
        theta_upper=cfg.theta_max + cfg.theta_cap
    )

    R_upper = R_upper_LU + N_F * E_upper
    R_lower = R_lower_LU + N_F * E_lower
    R_middle = (R_upper + R_lower) / 2.0

    return R_upper, R_lower, R_middle


def avg_rate_non_follower(rho_LU_total_dBW):
    """
    Non-follower方案的平均速率 (所有功率给单个Leader)
    """
    rho_total = 10**(rho_LU_total_dBW / 10)
    xi_lu_new = rho_total * cfg.G * cfg.zeta_U * (cfg.nu / (4 * np.pi))**2 / cfg.sigma_U_sq / 1e6
    return _avg_rate_leader_fast(xi_lu_new, pdf_theta_LU)


if __name__ == "__main__":
    # Test basic functions
    print("Testing core algorithm module...")
    print(f"theta_max = {np.rad2deg(cfg.theta_max):.4f} deg")
    print(f"theta_cap = {np.rad2deg(cfg.theta_cap):.4f} deg")

    # Test distance
    theta_test = 0.1  # rad
    print(f"r({theta_test:.2f}) = {distance_from_theta(theta_test):.2f} km")

    # Test PDF
    theta_arr = np.linspace(0.001, cfg.theta_max - 0.001, 100)
    pdf_vals = [pdf_theta_LU(t) for t in theta_arr]
    print(f"PDF integral = {np.trapezoid(pdf_vals, theta_arr):.6f} (should be 1.0)")

    # Test CDF
    print(f"F_W(1) = {cdf_W(1.0):.6f}")
    print(f"F_W(10) = {cdf_W(10.0):.6f}")

    # Test outage probability
    print(f"\nComputing Leader outage probability (gamma_th = {cfg.gamma_th_dB} dB)...")
    p_out = outage_leader()
    print(f"P_out_LU = {p_out:.6e}")
