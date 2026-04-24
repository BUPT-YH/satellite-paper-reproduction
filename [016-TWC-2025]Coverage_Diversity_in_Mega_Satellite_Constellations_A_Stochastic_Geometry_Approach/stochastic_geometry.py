"""
核心算法模块 - 随机几何分析
实现覆盖概率的解析模型:
1. 卫星选择分集 (Satellite Selection Diversity)
2. MRC 合并分集 (MRC Combining Diversity)

参考文献:
  Al Homssi et al., "Coverage Diversity in Mega Satellite Constellations:
  A Stochastic Geometry Approach," IEEE TWC, 2025.
"""

import numpy as np
from scipy import integrate
from scipy.special import erf
from scipy.stats import norm
import config as cfg


def alpha_m(Rm):
    """计算 α_m = R_⊕ / (R_⊕ + Rm), Eq.(2) 相关"""
    return cfg.R_EARTH / (cfg.R_EARTH + Rm)


def phi_max_m(Rm, psi_s=cfg.PSI_S, psi_t=cfg.PSI_T):
    """
    计算每壳层的最大地心角 φ_max, Eq.(2)-(3)
    考虑波束宽度约束和地平线约束
    """
    alpha = alpha_m(Rm)
    Rm_m = Rm * 1e3  # 转换为 m
    R_e = cfg.R_EARTH * 1e3  # 地球半径 m

    # 地平线波束宽度 ψ_o = 2*arcsin(α)
    psi_o = 2 * np.arcsin(alpha)

    # 有效波束宽度 ψ_m = min(ψ_s, 2*arcsin(1/α * sin(ψ_t/2))), Eq.(3)
    psi_m = min(psi_s, 2 * np.arcsin(np.sin(psi_t / 2) / alpha))

    if psi_m < psi_o:
        # 波束宽度约束情况, Eq.(2) 第一种
        phi_max = np.arcsin(np.sin(psi_m / 2) / alpha) - psi_m / 2
    else:
        # 地平线约束情况, Eq.(2) 第二种
        phi_max = np.arccos(alpha)

    return phi_max


def distance(phi, Rm):
    """
    计算卫星到地面用户的距离, Eq.(33)
    d(φ) = √(R²_⊕ + (R_⊕+Rm)² - 2*R_⊕*(R_⊕+Rm)*cos(φ))
    注意: Rm 单位为 km, 返回 m
    """
    R_e = cfg.R_EARTH * 1e3  # m
    R_sat = (cfg.R_EARTH + Rm) * 1e3  # m
    d2 = R_e**2 + R_sat**2 - 2 * R_e * R_sat * np.cos(phi)
    return np.sqrt(d2)


def path_loss(phi, Rm):
    """
    自由空间路径损耗 (FSPL) 幅度, 基于 Eq.(9)
    l(φ) = (c/(4π*fc*d(φ)))²
    返回线性值 (不是 dB)
    """
    d = distance(phi, Rm)  # m
    l = (cfg.C_LIGHT / (4 * np.pi * cfg.FC * d)) ** 2
    return l


def p_los(phi, Rm):
    """
    LoS 概率, Eq.(30)
    p_LoS(φ) = exp(-β*sin(φ)/(cos(φ) - α_m))
    当 cos(φ) <= α_m 时返回 0 (地平线以下)
    """
    alpha = alpha_m(Rm)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # 避免地平线以下的负值
    denominator = cos_phi - alpha
    result = np.where(denominator > 0,
                      np.exp(-cfg.BETA * sin_phi / denominator),
                      0.0)
    return result


def cdf_zeta(x, phi, Rm):
    """
    小尺度衰落的 CDF, Eq.(31)
    F_ζ(x) = 1/2 + p_LoS/2 * erf(...) + p_nLoS/2 * erf(...)
    x: 衰落幅度 (线性值, 不是 dB)
    phi: 地心角 (rad)
    Rm: 轨道高度 (km)
    """
    p_LoS = p_los(phi, Rm)
    p_nLoS = 1 - p_LoS

    # 将线性值转为 dB
    x_db = 10 * np.log10(np.maximum(x, 1e-30))

    # LoS 部分的 erf
    arg_los = (x_db + cfg.MU_LOS) / (np.sqrt(2) * cfg.SIGMA_LOS)
    # nLoS 部分的 erf
    arg_nlos = (x_db + cfg.MU_NLOS) / (np.sqrt(2) * cfg.SIGMA_NLOS)

    F = 0.5 + p_LoS / 2 * erf(arg_los) + p_nLoS / 2 * erf(arg_nlos)
    return np.clip(F, 0, 1)


def mean_zeta(phi, Rm):
    """
    计算平均小尺度衰落功率 ζ̄(φ)
    对对数正态混合分布取期望
    """
    p_LoS = p_los(phi, Rm)
    p_nLoS = 1 - p_LoS

    # 对数正态分布的均值: E[10^(X/10)] where X ~ N(μ, σ²)
    # = exp(μ*ln(10)/10 + σ²*(ln10/10)²/2)
    ln10_over_10 = np.log(10) / 10

    mu_los_lin = np.exp(cfg.MU_LOS * ln10_over_10 +
                        0.5 * (cfg.SIGMA_LOS * ln10_over_10) ** 2)
    mu_nlos_lin = np.exp(cfg.MU_NLOS * ln10_over_10 +
                         0.5 * (cfg.SIGMA_NLOS * ln10_over_10) ** 2)

    return p_LoS * mu_los_lin + p_nLoS * mu_nlos_lin


def avg_interference(Rm, lambda_u=cfg.LAMBDA_U):
    """
    计算每壳层的平均干扰, Eq.(16)
    Ī_m = 2π*λ*R²_⊕*ρ_t*G_t*G_r * ∫₀^φ_max l_m(φ)*ζ̄_m(φ)*sin(φ)dφ
    """
    phi_max = phi_max_m(Rm)
    R_e = cfg.R_EARTH * 1e3  # m

    def integrand(phi):
        l = path_loss(phi, Rm)
        zeta_bar = mean_zeta(phi, Rm)
        return l * zeta_bar * np.sin(phi)

    result, _ = integrate.quad(integrand, 0, phi_max)
    I_bar = (2 * np.pi * lambda_u * R_e**2 *
             cfg.RHO_T * cfg.GT * cfg.GR * result)
    return I_bar


def avg_n_satellites(Nm, Rm):
    """
    计算每壳层用户可视卫星平均数, Eq.(5)
    N̄_m = Nm/2 * (1 - cos(φ_max_m))
    """
    phi_max = phi_max_m(Rm)
    return Nm / 2 * (1 - np.cos(phi_max))


def pdf_B(phi, phi_max):
    """
    可视卫星方位角的 PDF, Eq.(6)
    f_B(φ) = sin(φ) / (1 - cos(φ_max))
    """
    return np.sin(phi) / (1 - np.cos(phi_max))


# ==================== 卫星选择分集 (Satellite Selection) ====================

def coverage_selection_single_shell(gamma_o, Nm, Rm, lambda_u=cfg.LAMBDA_U):
    """
    单壳层卫星选择分集覆盖概率, Lemma 3 + Eq.(20)
    P_SS(γo) = 1 - Σ_n exp(-N̄)/n! * [∫ F_ζ(...) f_B dφ]^n
    """
    phi_max = phi_max_m(Rm)
    N_bar = avg_n_satellites(Nm, Rm)
    I_bar = avg_interference(Rm, lambda_u)

    # 积分: P(SINR < γo) = ∫ F_ζ(γo*(Ī+Ws)/(ρ_t*G_t*G_r*l(φ))) * f_B(φ) dφ
    def integrand(phi):
        l = path_loss(phi, Rm)
        threshold = gamma_o * (I_bar + cfg.WS) / (cfg.RHO_T * cfg.GT * cfg.GR * l)
        F_zeta = cdf_zeta(threshold, phi, Rm)
        fb = pdf_B(phi, phi_max)
        return F_zeta * fb

    p_out_single, _ = integrate.quad(integrand, 1e-6, phi_max)

    # 对卫星数求和 (使用 Poisson 展开)
    # P_out = Σ_{n=0}^{∞} exp(-N̄)/n! * p_out_single^n
    # = exp(-N̄) * Σ_{n=0}^{∞} (N̄ * p_out_single)^n / n!
    # = exp(-N̄) * exp(N̄ * p_out_single)
    # = exp(-N̄ * (1 - p_out_single))
    P_coverage = 1 - np.exp(-N_bar * (1 - p_out_single))
    return P_coverage


def coverage_selection_multi_shell(gamma_o, Nm_list, Rm_list, lambda_u=cfg.LAMBDA_U):
    """
    多壳层卫星选择分集覆盖概率, Theorem 2, Eq.(22)
    P_SS(γo) = 1 - Π_m Σ_n exp(-N̄_m)/n! * [∫ F_ζ f_B dφ]^n
    """
    P_out_product = 1.0

    for m in range(len(Nm_list)):
        Nm = Nm_list[m]
        Rm = Rm_list[m]
        phi_max = phi_max_m(Rm)
        N_bar = avg_n_satellites(Nm, Rm)
        I_bar = avg_interference(Rm, lambda_u)

        def integrand(phi, Rm=Rm, I_bar=I_bar):
            l = path_loss(phi, Rm)
            threshold = gamma_o * (I_bar + cfg.WS) / (cfg.RHO_T * cfg.GT * cfg.GR * l)
            F_zeta = cdf_zeta(threshold, phi, Rm)
            fb = pdf_B(phi, phi_max)
            return F_zeta * fb

        p_out_single, _ = integrate.quad(integrand, 1e-6, phi_max)

        # 每壳层的 outage 概率
        P_out_m = np.exp(-N_bar * (1 - p_out_single))
        P_out_product *= P_out_m

    return 1 - P_out_product


# ==================== MRC 合并分集 (Combining Diversity) ====================

def _laplace_single_shell(s, Nm, Rm, lambda_u=cfg.LAMBDA_U):
    """
    单壳层 Laplace 变换, Eq.(45)
    L_m(s) = exp(-N_m/2 * ∫₀^φ_max (1 - E_ζ[exp(-s*ρ_t*G_t*G_r*l*ζ/(Ī+Ws))]) sin(φ) dφ)
    """
    phi_max = phi_max_m(Rm)
    I_bar = avg_interference(Rm, lambda_u)

    def integrand(phi):
        l = path_loss(phi, Rm)
        # 对 ζ 取期望: E_ζ[exp(-s * ρ_t*G_t*G_r*l*ζ/(Ī+Ws))]
        # 使用数值积分计算对数正态混合分布的期望
        a = s * cfg.RHO_T * cfg.GT * cfg.GR * l / (I_bar + cfg.WS)

        # 对 ζ 的 dB 空间 PDF 积分: E[exp(-a*ζ)] = ∫ exp(-a*10^(z/10)) * f_ζ_dB(z) dz
        # f_ζ_dB(z) = p_LoS/(σ_LoS√(2π))*exp(-(z+μ_LoS)²/(2σ²_LoS))
        #           + p_nLoS/(σ_nLoS√(2π))*exp(-(z+μ_nLoS)²/(2σ²_nLoS))
        def zeta_expectation(zeta_db):
            zeta_lin = 10 ** (zeta_db / 10)
            p_LoS_val = p_los(phi, Rm)
            p_nLoS_val = 1 - p_LoS_val

            # ζ[dB] 的 PDF (在 dB 空间直接积分，无需 1/zeta_lin 变换因子)
            pdf_los = (p_LoS_val / (cfg.SIGMA_LOS * np.sqrt(2 * np.pi)) *
                       np.exp(-(zeta_db + cfg.MU_LOS) ** 2 / (2 * cfg.SIGMA_LOS ** 2)))
            pdf_nlos = (p_nLoS_val / (cfg.SIGMA_NLOS * np.sqrt(2 * np.pi)) *
                        np.exp(-(zeta_db + cfg.MU_NLOS) ** 2 / (2 * cfg.SIGMA_NLOS ** 2)))
            pdf_zeta = pdf_los + pdf_nlos

            val = np.exp(-a * zeta_lin) * pdf_zeta
            return val

        E_zeta, _ = integrate.quad(zeta_expectation, -30, 20,
                                   limit=100, epsabs=1e-10)
        return (1 - E_zeta) * np.sin(phi)

    result, _ = integrate.quad(integrand, 1e-6, phi_max, limit=100)
    L = np.exp(-Nm / 2 * result)
    return L


def coverage_combining_single_shell(gamma_o, Nm, Rm, lambda_u=cfg.LAMBDA_U, L=cfg.L_EULER):
    """
    单壳层合并分集覆盖概率, Lemma 4, Eq.(25)
    使用 Euler 反演公式计算 CDF 的逆 Laplace 变换
    """
    phi_max = phi_max_m(Rm)
    I_bar = avg_interference(Rm, lambda_u)

    # 计算该 gamma 对应的 Laplace 变换
    def laplace_at(s):
        return _laplace_single_shell(s, Nm, Rm, lambda_u)

    # Euler 反演: F(x) = (10^(L/3)/x) * Σ_{l=0}^{2L} (-1)^l * ξ_l * Re(x/β_l * L(β_l/x))
    total = 0.0
    for l_val in range(2 * L + 1):
        beta_l = L * np.log(10) / 3 + 1j * np.pi * l_val

        # 计算 ξ_l 系数
        if l_val == 0:
            xi_l = 0.5
        elif l_val == 2 * L:
            xi_l = 1.0 / (2 * L)
        elif l_val == 2 * L - 1:
            xi_l = 1.0 / (2 * L)
        elif l_val <= L:
            xi_l = 1.0
        else:
            # ξ_{2L-l} = ξ_{2L-l+1} + 2^{-L} * C(L,l)
            # 递推计算
            xi_l = _compute_xi(l_val, L)

        s_val = beta_l / gamma_o
        L_val = laplace_at(s_val)
        term = (-1) ** l_val * xi_l * np.real(gamma_o / beta_l * L_val)
        total += term

    F = (10 ** (L / 3) / gamma_o) * total
    return 1 - np.real(F)


def coverage_combining_multi_shell(gamma_o, Nm_list, Rm_list,
                                    lambda_u=cfg.LAMBDA_U, L=cfg.L_EULER):
    """
    多壳层合并分集覆盖概率, Eq.(48)-(49)
    L_Σ(s) = Π_m L_m(s), 然后 Euler 反演
    """
    # 计算总 Laplace 变换
    def laplace_total(s):
        L_total = 1.0
        for m in range(len(Nm_list)):
            L_total *= _laplace_single_shell(s, Nm_list[m], Rm_list[m], lambda_u)
        return L_total

    # Euler 反演
    total = 0.0
    for l_val in range(2 * L + 1):
        beta_l = L * np.log(10) / 3 + 1j * np.pi * l_val

        if l_val == 0:
            xi_l = 0.5
        elif l_val == 2 * L:
            xi_l = 1.0 / (2 * L)
        elif l_val <= L:
            xi_l = 1.0
        else:
            xi_l = _compute_xi(l_val, L)

        s_val = beta_l / gamma_o
        L_val = laplace_total(s_val)
        term = (-1) ** l_val * xi_l * np.real(gamma_o / beta_l * L_val)
        total += term

    F = (10 ** (L / 3) / gamma_o) * total
    return 1 - np.real(F)


def _compute_xi(l, L):
    """
    计算 Euler 反演系数 ξ_l
    ξ_0 = 1/2, ξ_l = 1 for 1<=l<=L, ξ_{2L} = 1/(2L)
    ξ_{2L-l} = ξ_{2L-l+1} + 2^{-L} * C(L,l) for 0 < l < L
    """
    from math import comb
    # 从 ξ_{2L} 递推回 ξ_l
    xi = {}
    xi[2 * L] = 1.0 / (2 * L)
    for j in range(1, L + 1):
        idx = 2 * L - j
        xi[idx] = xi[idx + 1] + 2 ** (-L) * comb(L, j)
    return xi.get(l, 1.0)


# ==================== Monte Carlo 仿真 ====================

def monte_carlo_selection(gamma_o, Nm_list, Rm_list, lambda_u=cfg.LAMBDA_U,
                          n_samples=cfg.N_MC_SAMPLES):
    """
    Monte Carlo 仿真: 卫星选择分集
    对每个用户随机生成可视卫星位置, 选择最佳 SINR
    """
    M = len(Nm_list) if isinstance(Nm_list, list) else 1
    if M == 1:
        Nm_list = [Nm_list]
        Rm_list = [Rm_list]

    n_covered = 0

    for _ in range(n_samples):
        best_sinr = -np.inf

        for m in range(M):
            Nm = Nm_list[m]
            Rm = Rm_list[m]
            phi_max = phi_max_m(Rm)
            I_bar = avg_interference(Rm, lambda_u)

            # 随机生成该壳层中可视卫星数 (Poisson 近似)
            N_bar = avg_n_satellites(Nm, Rm)
            n_sat = np.random.poisson(N_bar)

            if n_sat == 0:
                continue

            # 随机生成卫星位置 (地心角)
            # f_B(φ) = sin(φ)/(1-cos(φ_max)), 用 CDF 反演
            u = np.random.uniform(0, 1, n_sat)
            cos_phi_max = np.cos(phi_max)
            phis = np.arccos(1 - u * (1 - cos_phi_max))

            for phi in phis:
                l = path_loss(phi, Rm)
                # 生成随机小尺度衰落
                p_LoS_val = p_los(phi, Rm)
                if np.random.uniform() < p_LoS_val:
                    zeta_db = np.random.normal(-cfg.MU_LOS, cfg.SIGMA_LOS)
                else:
                    zeta_db = np.random.normal(-cfg.MU_NLOS, cfg.SIGMA_NLOS)
                zeta = 10 ** (zeta_db / 10)

                # 信号功率
                S = cfg.RHO_T * cfg.GT * cfg.GR * l * zeta
                sinr = S / (I_bar + cfg.WS)
                best_sinr = max(best_sinr, sinr)

        if best_sinr >= gamma_o:
            n_covered += 1

    return n_covered / n_samples


def monte_carlo_combining(gamma_o, Nm_list, Rm_list, lambda_u=cfg.LAMBDA_U,
                          n_samples=cfg.N_MC_SAMPLES):
    """
    Monte Carlo 仿真: MRC 合并分集
    对每个用户随机生成所有可视卫星位置, 合并所有 SINR
    """
    M = len(Nm_list) if isinstance(Nm_list, list) else 1
    if M == 1:
        Nm_list = [Nm_list]
        Rm_list = [Rm_list]

    n_covered = 0

    for _ in range(n_samples):
        # 论文模型: SINR = Σ_m Σ_n S_{m,n} / (Ī_m + Ws)
        # 每个壳层的信号除以各自壳层的平均干扰+噪声
        total_sinr = 0.0

        for m in range(M):
            Nm = Nm_list[m]
            Rm = Rm_list[m]
            phi_max = phi_max_m(Rm)
            I_bar = avg_interference(Rm, lambda_u)

            N_bar = avg_n_satellites(Nm, Rm)
            n_sat = np.random.poisson(N_bar)

            if n_sat == 0:
                continue

            u = np.random.uniform(0, 1, n_sat)
            cos_phi_max = np.cos(phi_max)
            phis = np.arccos(1 - u * (1 - cos_phi_max))

            for phi in phis:
                l = path_loss(phi, Rm)
                p_LoS_val = p_los(phi, Rm)
                if np.random.uniform() < p_LoS_val:
                    zeta_db = np.random.normal(-cfg.MU_LOS, cfg.SIGMA_LOS)
                else:
                    zeta_db = np.random.normal(-cfg.MU_NLOS, cfg.SIGMA_NLOS)
                zeta = 10 ** (zeta_db / 10)

                # MRC 合并: 按论文模型, 每个卫星信号除以该壳层 Ī_m + Ws
                S = cfg.RHO_T * cfg.GT * cfg.GR * l * zeta
                total_sinr += S / (I_bar + cfg.WS)

        if total_sinr >= gamma_o:
            n_covered += 1

    return n_covered / n_samples
