"""
核心仿真模块: Monte Carlo仿真 + 解析表达式
论文: Downlink Performance of Cell-Free Massive MIMO for LEO Satellite Mega-Constellation
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
from scipy.special import hyp1f1
from config import (get_params, compute_distance_bounds, compute_Hv,
                    compute_avg_ut_number, freq_factor, c_km)

# ============================================================
# 通用工具
# ============================================================

def generate_sap_positions(RS, lambda_S, rng, N_total=None):
    """在半径RS的球面上生成PPP分布的SAP位置"""
    area = 4 * np.pi * RS**2
    if N_total is None:
        N_total = rng.poisson(lambda_S * area)
    if N_total == 0:
        return np.empty((0, 3))
    phi = rng.uniform(0, 2 * np.pi, N_total)
    cos_theta = rng.uniform(-1, 1, N_total)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x = RS * sin_theta * np.cos(phi)
    y = RS * sin_theta * np.sin(phi)
    z = RS * cos_theta
    return np.column_stack([x, y, z])


def generate_starlink_constellation(RS, inclinations, planes_per_inc, sats_per_plane,
                                     rng=None, random_state=False):
    """生成Walker星座 (用于Fig.4 Starlink对比)
    random_state: True=随机初始RAAN和相位, False=固定均匀分布
    """
    all_pos = []
    for inc_deg in inclinations:
        inc = np.deg2rad(inc_deg)
        for p in range(planes_per_inc):
            if random_state and rng is not None:
                # 随机初始状态: RAAN完全随机, 初始相位随机
                raan = rng.uniform(0, 2 * np.pi)
                phase_offset = rng.uniform(0, 2 * np.pi)
            else:
                # 固定初始状态: 均匀分布RAAN
                raan = 2 * np.pi * p / planes_per_inc
                phase_offset = 0
            for n in range(sats_per_plane):
                arg_lat = 2 * np.pi * n / sats_per_plane + phase_offset
                cos_raan, sin_raan = np.cos(raan), np.sin(raan)
                cos_inc, sin_inc = np.cos(inc), np.sin(inc)
                cos_a, sin_a = np.cos(arg_lat), np.sin(arg_lat)
                x_orb = RS * cos_a
                y_orb = RS * sin_a
                x = (cos_raan * x_orb - sin_raan * cos_inc * y_orb)
                y = (sin_raan * x_orb + cos_raan * cos_inc * y_orb)
                z = sin_inc * y_orb
                all_pos.append([x, y, z])
    return np.array(all_pos)


def nakagami_sq(m, size, rng):
    """生成 |h|² ~ Gamma(m, 1/m), E[|h|²] = 1"""
    return rng.gamma(m, 1.0 / m, size=size)


# ============================================================
# Monte Carlo 仿真引擎
# ============================================================

def mc_sinr_single(p, rng, use_lmmse=False, tau_p_val=200):
    """
    单次Monte Carlo仿真, 返回典型UT的SINR(线性值)
    典型UT位于北极 (0, 0, RE)
    """
    RS, RE = p['RS'], p['RE']
    m = p['m']
    alpha_val = p['alpha']
    Gml_val = p['Gml']
    Gsl_val = p['Gsl']
    rho_d_val = p['rho_d']
    sigma2_val = p['sigma2']
    eta = p['eta']

    rS_min, rS_max, rmax = compute_distance_bounds(p)
    ut_pos = np.array([0.0, 0.0, RE])

    # 生成SAP
    sap_pos = generate_sap_positions(RS, p['lambda_S'], rng)
    if len(sap_pos) == 0:
        return -np.inf, 0

    # 计算距离
    distances = np.linalg.norm(sap_pos - ut_pos, axis=1)

    # 分类: 服务SAP / 干扰SAP
    service_mask = (distances >= rS_min) & (distances <= rS_max)
    interf_mask = (distances > rS_max) & (distances <= rmax)

    N_ser = np.sum(service_mask)
    N_int = np.sum(interf_mask)

    if N_ser == 0:
        return -np.inf, N_ser

    # 平均每SAP服务UT数
    phi_U_avg = compute_avg_ut_number(p)

    # 路径损耗
    d_ser = distances[service_mask]
    beta_ser = d_ser**(-alpha_val)

    # 生成Nakagami-m信道 (|h|²)
    h2_ser = nakagami_sq(m, N_ser, rng)
    h_ser = np.sqrt(h2_ser)
    phase_ser = rng.uniform(0, 2 * np.pi, N_ser)
    h_complex_ser = h_ser * np.exp(1j * phase_ser)

    if use_lmmse:
        # LMMSE信道估计 (Eq.6)
        # 导频SNR: tau_p * rho_p * Gml * beta / sigma^2
        rho_p_val = p.get('rho_p', p['rho_d'])
        snr_pilot = tau_p_val * rho_p_val * Gml_val * beta_ser / sigma2_val
        est_var = snr_pilot / (1 + snr_pilot)
        # 估计信道 = sqrt(est_var) * h + sqrt(1-est_var) * noise
        noise_re = rng.standard_normal(N_ser)
        noise_im = rng.standard_normal(N_ser)
        noise_est = (noise_re + 1j * noise_im) / np.sqrt(2) * np.sqrt(1 - est_var)
        h_hat_ser = np.sqrt(est_var) * h_complex_ser + noise_est
        h_hat_mag = np.abs(h_hat_ser)
        h_hat_mag = np.maximum(h_hat_mag, 1e-10)
    else:
        h_hat_ser = h_complex_ser
        h_hat_mag = h_ser

    # 期望信号 DS
    # DS = Σ_l sqrt(ρd/|ΦU| × Gml) × β^(1/2) × h × ĥ*/|ĥ|
    bf_dir = h_hat_ser.conj() / h_hat_mag
    DS = np.sum(np.sqrt(rho_d_val / phi_U_avg * Gml_val) * np.sqrt(beta_ser)
                * h_complex_ser * bf_dir)

    # 多用户干扰 MUI
    N_other = max(0, int(round(phi_U_avg)) - 1)
    MUI = 0.0 + 0.0j
    if N_other > 0:
        # 每个服务SAP有N_other个其他UT, 每个有独立信道
        h2_other = nakagami_sq(m, (N_ser, N_other), rng)
        h_other = np.sqrt(h2_other)
        phase_other = rng.uniform(0, 2 * np.pi, (N_ser, N_other))
        h_other_c = h_other * np.exp(1j * phase_other)

        if use_lmmse:
            rho_p_val = p.get('rho_p', p['rho_d'])
            snr_pilot_other = tau_p_val * rho_p_val * Gml_val * beta_ser[:, None] / sigma2_val
            est_var_other = snr_pilot_other / (1 + snr_pilot_other)
            noise_other_re = rng.standard_normal((N_ser, N_other))
            noise_other_im = rng.standard_normal((N_ser, N_other))
            noise_est_other = (noise_other_re + 1j * noise_other_im) / np.sqrt(2) * np.sqrt(1 - est_var_other)
            h_hat_other = np.sqrt(est_var_other) * h_other_c + noise_est_other
            h_hat_other_mag = np.abs(h_hat_other)
            h_hat_other_mag = np.maximum(h_hat_other_mag, 1e-10)
            bf_other = h_hat_other.conj() / h_hat_other_mag
        else:
            bf_other = h_other_c.conj() / h_other

        q_other = (rng.standard_normal((N_ser, N_other))
                   + 1j * rng.standard_normal((N_ser, N_other))) / np.sqrt(2)
        mui_per_sap = np.sqrt(rho_d_val / phi_U_avg * Gml_val) * np.sqrt(beta_ser) \
                      * h_complex_ser * np.sum(bf_other * q_other, axis=1)
        MUI = np.sum(mui_per_sap)

    # 星间干扰 ISI
    ISI = 0.0 + 0.0j
    if N_int > 0:
        d_int = distances[interf_mask]
        beta_int = d_int**(-alpha_val)
        h2_int = nakagami_sq(m, N_int, rng)
        h_int = np.sqrt(h2_int)
        phase_int = rng.uniform(0, 2 * np.pi, N_int)
        h_int_c = h_int * np.exp(1j * phase_int)
        q_int = (rng.standard_normal(N_int) + 1j * rng.standard_normal(N_int)) / np.sqrt(2)
        ISI = np.sum(np.sqrt(rho_d_val * Gsl_val) * np.sqrt(beta_int) * h_int_c * q_int)

    # SINR
    signal_power = np.abs(DS)**2
    interference_power = np.abs(MUI)**2 + np.abs(ISI)**2
    sinr = signal_power / (interference_power + sigma2_val)
    return sinr, N_ser


def mc_sinr_single_cell_based(p, rng):
    """Cell-based方案: 典型UT仅由最近SAP服务 (用于Fig.7对比)
    ISI仅来自服务穹顶外的干扰SAP (rS_max到rmax)
    """
    RS, RE = p['RS'], p['RE']
    m = p['m']
    alpha_val = p['alpha']
    Gml_val = p['Gml']
    Gsl_val = p['Gsl']
    rho_d_val = p['rho_d']
    sigma2_val = p['sigma2']
    rS_min, rS_max, rmax = compute_distance_bounds(p)

    ut_pos = np.array([0.0, 0.0, RE])
    sap_pos = generate_sap_positions(RS, p['lambda_S'], rng)
    if len(sap_pos) == 0:
        return -np.inf

    distances = np.linalg.norm(sap_pos - ut_pos, axis=1)
    valid_mask = (distances >= rS_min) & (distances <= rmax)
    if not np.any(valid_mask):
        return -np.inf

    valid_dist = distances[valid_mask]
    nearest_idx = np.argmin(valid_dist)
    d_nearest = valid_dist[nearest_idx]

    beta_nearest = d_nearest**(-alpha_val)
    h2 = nakagami_sq(m, 1, rng)[0]
    h = np.sqrt(h2)

    DS = np.sqrt(rho_d_val * Gml_val) * np.sqrt(beta_nearest) * h

    # ISI: 仅来自穹顶外的干扰SAP (rS_max到rmax范围)
    interf_mask = (distances > rS_max) & (distances <= rmax)
    ISI = 0.0
    if np.any(interf_mask):
        d_other = distances[interf_mask]
        beta_other = d_other**(-alpha_val)
        h2_other = nakagami_sq(m, len(beta_other), rng)
        ISI = np.sum(rho_d_val * Gsl_val * beta_other * h2_other)

    sinr = np.abs(DS)**2 / (ISI + sigma2_val)
    return sinr


def mc_coverage(p, gamma_th_range_dB, n_realizations=5000, seed=42,
                cell_based=False, use_lmmse=False, tau_p_val=200):
    """Monte Carlo覆盖概率"""
    rng = np.random.default_rng(seed)
    gamma_th_lin = 10**(np.array(gamma_th_range_dB) / 10)
    n_thresholds = len(gamma_th_range_dB)
    sinr_samples = np.zeros(n_realizations)

    for i in range(n_realizations):
        if cell_based:
            sinr = mc_sinr_single_cell_based(p, rng)
        else:
            sinr, _ = mc_sinr_single(p, rng, use_lmmse=use_lmmse, tau_p_val=tau_p_val)
        sinr_samples[i] = sinr if np.isfinite(sinr) else -1e10

    coverage = np.zeros(n_thresholds)
    for j, gth in enumerate(gamma_th_lin):
        coverage[j] = np.mean(sinr_samples > gth)
    return coverage


def mc_dss_ccdf(p, x_range, n_realizations=10000, seed=42,
                use_lmmse=False, tau_p_val=200):
    """Monte Carlo仿真DSS的CCDF"""
    rng = np.random.default_rng(seed)
    dss_samples = np.zeros(n_realizations)

    for i in range(n_realizations):
        RS, RE = p['RS'], p['RE']
        m = p['m']
        alpha_val = p['alpha']
        eta = p['eta']
        rS_min, rS_max, rmax = compute_distance_bounds(p)
        ut_pos = np.array([0.0, 0.0, RE])
        sap_pos = generate_sap_positions(RS, p['lambda_S'], rng)
        if len(sap_pos) == 0:
            dss_samples[i] = 0
            continue
        distances = np.linalg.norm(sap_pos - ut_pos, axis=1)
        service_mask = (distances >= rS_min) & (distances <= rS_max)
        N_ser = np.sum(service_mask)
        if N_ser == 0:
            dss_samples[i] = 0
            continue

        phi_U_avg = compute_avg_ut_number(p)
        d_ser = distances[service_mask]
        beta_ser = d_ser**(-alpha_val)
        h2_ser = nakagami_sq(m, N_ser, rng)
        h_ser = np.sqrt(h2_ser)

        if use_lmmse:
            rho_p_val = p.get('rho_p', p['rho_d'])
            sigma2_val = p['sigma2']
            # 导频SNR需包含天线增益Gml (Eq.6)
            Gml_val = p['Gml']
            snr_pilot = tau_p_val * rho_p_val * Gml_val * beta_ser / sigma2_val
            est_var = snr_pilot / (1 + snr_pilot)
            noise_re = rng.standard_normal(N_ser)
            noise_im = rng.standard_normal(N_ser)
            noise_est = (noise_re + 1j * noise_im) / np.sqrt(2) * np.sqrt(1 - est_var)
            phase_ser = rng.uniform(0, 2 * np.pi, N_ser)
            h_complex_ser = h_ser * np.exp(1j * phase_ser)
            h_hat_ser = np.sqrt(est_var) * h_complex_ser + noise_est
            h_hat_mag = np.abs(h_hat_ser)
            h_hat_mag = np.maximum(h_hat_mag, 1e-10)
            # DS strength
            DS = np.abs(np.sum(np.sqrt(p['rho_d'] / phi_U_avg * p['Gml'])
                               * np.sqrt(beta_ser) * h_complex_ser * h_hat_ser.conj() / h_hat_mag))
        else:
            # DSS = Σ_l sqrt(ρd/|ΦU| × Gml) × β^(1/2) × |h|
            DS = np.sum(np.sqrt(p['rho_d'] / phi_U_avg * p['Gml']) * np.sqrt(beta_ser) * h_ser)

        dss_samples[i] = DS

    ccdf = np.zeros(len(x_range))
    for j, x in enumerate(x_range):
        ccdf[j] = np.mean(dss_samples > x)
    return ccdf, dss_samples


# ============================================================
# 解析表达式
# ============================================================

def compute_E_h_exp_numerical(s_complex, beta_val, m):
    """数值计算 E_h[exp(-s × √β × |h|)], |h| ~ Nakagami(m, Ω=1)
    使用积分: ∫₀^∞ exp(-t×x) × 2m^m/Γ(m) × x^{2m-1} × exp(-mx²) dx
    其中 t = s × √β
    """
    t = s_complex * np.sqrt(beta_val)

    def integrand_real(x):
        return (np.exp(-t.real * x) * np.cos(t.imag * x)
                * 2 * m**m / gamma_func(m) * x**(2*m-1) * np.exp(-m * x**2))

    def integrand_imag(x):
        return (-np.exp(-t.real * x) * np.sin(t.imag * x)
                * 2 * m**m / gamma_func(m) * x**(2*m-1) * np.exp(-m * x**2))

    re, _ = quad(integrand_real, 0, 200, limit=200)
    im, _ = quad(integrand_imag, 0, 200, limit=200)
    return re + 1j * im


def laplace_S_hat(s_complex, p):
    """计算 Ŝ_k 的 Laplace 变换 (Eq.24)
    L_Ŝ(s) = exp(-2πλS × RS/RE × ∫[r_min,r_max] r×(1 - g(s,r)) dr)
    """
    RS, RE = p['RS'], p['RE']
    lambda_S = p['lambda_S']
    m = p['m']
    alpha_val = p['alpha']
    rS_min, rS_max, _ = compute_distance_bounds(p)

    def integrand_r_re(r):
        beta_val = r**(-alpha_val)
        g = compute_E_h_exp_numerical(s_complex, beta_val, m)
        return r * (1 - g.real)

    def integrand_r_im(r):
        beta_val = r**(-alpha_val)
        g = compute_E_h_exp_numerical(s_complex, beta_val, m)
        return r * (-g.imag)

    int_re, _ = quad(integrand_r_re, rS_min, rS_max, limit=100)
    int_im, _ = quad(integrand_r_im, rS_min, rS_max, limit=100)

    exponent = -2 * np.pi * lambda_S * (RS / RE) * (int_re + 1j * int_im)
    return np.exp(exponent)


def cdf_S_hat_numerical(x, p, A=20, B=15, C=15):
    """数值计算 Ŝ_k 的 CDF (Eq.25)
    F(x) ≈ 2^(-B) × exp(A/2) × Σ_{b=0}^{B} Σ_{c=0}^{C+b} ...
    """
    if x <= 0:
        return 0.0

    result = 0.0
    for b in range(B + 1):
        for c in range(C + b + 1):
            s = (A + 1j * 2 * np.pi * c) / (2 * x)
            D_c = 2 if c == 0 else 1
            coeff = ((-1)**c) * D_c
            from scipy.special import comb
            try:
                comb_b = comb(B, b, exact=True)
            except Exception:
                comb_b = float(comb(B, b))
            L_s = laplace_S_hat(s, p)
            val = (L_s / s)
            result += coeff * comb_b * val.real

    result *= 2**(-B) * np.exp(A / 2) * x
    return min(max(result, 0.0), 1.0)


def avg_mui(p):
    """平均MUI (Eq.27): E[|h|²]=1, 不依赖m"""
    RS, RE = p['RS'], p['RE']
    lambda_S = p['lambda_S']
    alpha_val = p['alpha']
    rS_min, rS_max, _ = compute_distance_bounds(p)
    phi_U_avg = compute_avg_ut_number(p)

    coeff = 2 * np.pi * lambda_S * RS / RE * (phi_U_avg - 1) / phi_U_avg
    if alpha_val == 2:
        integral = np.log(rS_max / rS_min)
    else:
        integral = (rS_max**(2 - alpha_val) - rS_min**(2 - alpha_val)) / (2 - alpha_val)
    return coeff * integral


def avg_isi(p):
    """平均ISI (Eq.28): E[|h|²]=1, 不依赖m"""
    RS, RE = p['RS'], p['RE']
    lambda_S = p['lambda_S']
    alpha_val = p['alpha']
    Gml_val = p['Gml']
    Gsl_val = p['Gsl']
    _, rS_max, rmax = compute_distance_bounds(p)

    coeff = (Gsl_val / Gml_val) * 2 * np.pi * lambda_S * RS / RE
    if alpha_val == 2:
        integral = np.log(rmax / rS_max)
    else:
        integral = (rmax**(2 - alpha_val) - rS_max**(2 - alpha_val)) / (2 - alpha_val)
    return coeff * integral


def coverage_analytical(gamma_th_dB, p, A=20, B=15, C=15):
    """解析覆盖概率 (Theorem 1, Eq.31)
    覆盖条件: Ŝ_k ≥ √(γ_th × |ΦU|_avg × (ISer + IInt + σ²/(ρd×Gml)))
    """
    gamma_th = 10**(gamma_th_dB / 10)
    phi_U_avg = compute_avg_ut_number(p)
    if phi_U_avg <= 0:
        return 0.0

    ISer = avg_mui(p)
    IInt = avg_isi(p)
    rho_d_val = p['rho_d']
    Gml_val = p['Gml']
    sigma2_val = p['sigma2']

    threshold = np.sqrt(phi_U_avg
                        * gamma_th * (ISer + IInt + sigma2_val / (rho_d_val * Gml_val)))

    cdf_val = cdf_S_hat_numerical(threshold, p, A=A, B=B, C=C)
    return 1.0 - cdf_val


def coverage_analytical_fast(gamma_th_range_dB, p, **kwargs):
    """批量计算解析覆盖概率"""
    return np.array([coverage_analytical(g, p, **kwargs) for g in gamma_th_range_dB])


# ============================================================
# 容量计算
# ============================================================

def capacity_cf_system(N_U, p, gamma_max_dB=30, n_points=500):
    """CF方案系统容量 (Eq.34)"""
    gamma_range = np.linspace(0, gamma_max_dB, n_points)
    p_cov = coverage_analytical_fast(gamma_range, p)
    # E[log2(1+SINR)] = ∫₀^∞ P_cov(2^t - 1) dt (Eq.33)
    t_range = np.linspace(0, gamma_max_dB / 10 * np.log2(10), n_points)
    gamma_th_from_t = 10 * np.log10(2**t_range - 1 + 1e-30)
    # 重新计算覆盖概率 vs t
    sinr_thresholds = 10**(gamma_th_from_t / 10)
    # 使用 dB 范围重新插值
    from numpy import interp
    p_cov_interp = interp(gamma_th_from_t, gamma_range, p_cov)
    spectral_eff = np.trapz(p_cov_interp, t_range)
    pre_factor = (p['tau_c'] - p['tau_p']) / p['tau_c']
    return N_U * p['B'] * pre_factor * spectral_eff


def capacity_nearest_system(N_U, p, gamma_max_dB=30, n_points=500):
    """最近卫星方案系统容量 (Eq.35)"""
    gamma_range = np.linspace(0, gamma_max_dB, n_points)
    p_cov = coverage_analytical_fast(gamma_range, p)
    t_range = np.linspace(0, gamma_max_dB / 10 * np.log2(10), n_points)
    gamma_th_from_t = 10 * np.log10(2**t_range - 1 + 1e-30)
    from numpy import interp
    p_cov_interp = interp(gamma_th_from_t, gamma_range, p_cov)
    spectral_eff = np.trapz(p_cov_interp, t_range)
    return N_U * p['B'] * spectral_eff


# ============================================================
# 无波束赋形覆盖概率
# ============================================================

def coverage_no_bf_mc(gamma_th_dB, p, n_mc=5000, seed=42):
    """无波束赋形时的覆盖概率 (Monte Carlo)
    DS: 非相干功率和 sum(rho_d/phi_U * Gml * beta * |h|^2)
    MUI: 独立信道功率和 (不与DS共享同一信道实现)
    ISI: 穹顶外干扰SAP的功率和
    """
    rng = np.random.default_rng(seed)
    gamma_th_lin = 10**(gamma_th_dB / 10)
    sinr_samples = np.zeros(n_mc)

    RS, RE = p['RS'], p['RE']
    m_val = p['m']
    alpha_val = p['alpha']
    Gml_val = p['Gml']
    Gsl_val = p['Gsl']
    rho_d_val = p['rho_d']
    sigma2_val = p['sigma2']
    rS_min, rS_max, rmax = compute_distance_bounds(p)
    ut_pos = np.array([0.0, 0.0, RE])
    phi_U_avg = compute_avg_ut_number(p)

    for i in range(n_mc):
        sap_pos = generate_sap_positions(RS, p['lambda_S'], rng)
        if len(sap_pos) == 0:
            sinr_samples[i] = -1e10
            continue
        distances = np.linalg.norm(sap_pos - ut_pos, axis=1)
        service_mask = (distances >= rS_min) & (distances <= rS_max)
        interf_mask = (distances > rS_max) & (distances <= rmax)

        if np.any(service_mask):
            d_ser = distances[service_mask]
            beta_ser = d_ser**(-alpha_val)
            h2_ser = nakagami_sq(m_val, len(d_ser), rng)
            # 无BF: 信号功率 = Σ ρd/|ΦU| × Gml × β × |h|²
            signal_power = np.sum(rho_d_val / phi_U_avg * Gml_val * beta_ser * h2_ser)
        else:
            signal_power = 0

        # MUI: 使用解析平均值的随机波动
        interf_power = avg_mui(p) * rho_d_val * Gml_val * rng.gamma(1, 1)

        if np.any(interf_mask):
            d_int = distances[interf_mask]
            beta_int = d_int**(-alpha_val)
            h2_int = nakagami_sq(m_val, len(d_int), rng)
            interf_power += np.sum(rho_d_val * Gsl_val * beta_int * h2_int)

        sinr_samples[i] = signal_power / (interf_power + sigma2_val)

    return np.mean(sinr_samples > gamma_th_lin)
