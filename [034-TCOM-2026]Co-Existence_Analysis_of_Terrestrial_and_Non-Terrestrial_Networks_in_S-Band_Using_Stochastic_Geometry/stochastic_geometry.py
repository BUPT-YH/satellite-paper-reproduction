"""
stochastic_geometry.py — 核心随机几何计算模块
论文: Co-Existence Analysis of Terrestrial and Non-Terrestrial Networks
      in S-Band Using Stochastic Geometry (IEEE TCOM 2026)

实现所有CDF、PDF、Laplace变换、覆盖概率的计算函数。

重要: 距离分布(CDF/PDF)使用km单位，路径损耗计算中使用m单位。
      路径损耗模型 PL(d) = d^{-alpha} 中d以m为单位，这样NTN干扰
      (卫星高度200-1200km)与TN干扰(百米量级)的相对大小才合理。

数值精度说明:
  内层积分(TN干扰Laplace、NTN上行Laplace)使用CDF差分(Riemann-Stieltjes)
  替代数值PDF微分，避免在边界处的微分误差被n^active或n_u放大。
"""

import numpy as np
from scipy.integrate import quad
from config import (
    R_EARTH, ALPHA_TN, ALPHA_NTN, M_TN, M_NTN,
    G_TN, G_NTN_DL, G_NTN_UL, P_TN, P_NTN_DL, P_NTN_UL,
    SIGMA2, N_C, QUAD_EPSABS, QUAD_EPSREL, QUAD_LIMIT,
    R_NTN,
)

# km到m的转换因子
KM2M = 1000.0


# ============================================================
# 辅助函数: Nakagami-m 信道的 Laplace 变换
# ============================================================
def nakagami_laplace(s, p, g, r_m, alpha, m):
    """
    Nakagami-m 信道的 Laplace 变换
    L_H(s*p*G*r^{-alpha}) = (1 + s*p*g*r^{-alpha}/m)^{-m}

    参数:
        s: Laplace 参数
        p: 发射功率 (W)
        g: 等效天线增益 (线性)
        r_m: 距离 (m) — 必须用米
        alpha: 路径损耗指数
        m: Nakagami-m 参数
    """
    if r_m <= 0:
        return 0.0
    arg = s * p * g * r_m ** (-alpha) / m
    if arg < 1e-15:
        return 1.0
    if arg > 500:
        return 0.0
    return (1.0 + arg) ** (-m)


# ============================================================
# 服务距离分布 (公式3) — 距离单位: km
# ============================================================
def single_point_cdf(r_0, r_TN, x_0):
    """
    单个BS距离的CDF: F_R(r_0)
    UE位于(x_0, 0)，BS均匀分布在半径r_TN的圆内
    """
    if r_0 < 0:
        return 0.0
    if x_0 == 0:
        if r_0 <= r_TN:
            return (r_0 / r_TN) ** 2
        else:
            return 1.0

    if r_0 <= max(0, r_TN - x_0):
        return (r_0 / r_TN) ** 2
    elif r_0 <= r_TN + x_0:
        val = r_0 ** 2 + x_0 ** 2 - r_TN ** 2
        cos_theta = val / (2 * x_0 * r_0)
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)

        cos_phi = (r_TN ** 2 + x_0 ** 2 - r_0 ** 2) / (2 * x_0 * r_TN)
        cos_phi = np.clip(cos_phi, -1, 1)
        phi = np.arccos(cos_phi)

        area = r_0 ** 2 * theta + r_TN ** 2 * phi - x_0 * r_0 * np.sin(theta)
        total_area = np.pi * r_TN ** 2
        return area / total_area
    else:
        return 1.0


def serving_distance_cdf(r_0, r_TN, x_0, n_c):
    """
    服务距离CDF: F_{R_0}(r_0) = 1 - (1 - F_R(r_0))^{N_c}  (公式3)
    """
    if r_0 <= 0:
        return 0.0
    if r_0 >= r_TN + x_0:
        return 1.0
    f_r = single_point_cdf(r_0, r_TN, x_0)
    return 1.0 - (1.0 - f_r) ** n_c


def serving_distance_pdf(r_0, r_TN, x_0, n_c, dx_km=1e-4):
    """
    服务距离PDF: f_{R_0}(r_0) = d/dr_0 F_{R_0}(r_0)
    使用中心差分, dx_km为差分步长 (km)
    """
    r_minus = max(0, r_0 - dx_km)
    r_plus = r_0 + dx_km
    cdf_plus = serving_distance_cdf(r_plus, r_TN, x_0, n_c)
    cdf_minus = serving_distance_cdf(r_minus, r_TN, x_0, n_c)
    return (cdf_plus - cdf_minus) / (2 * dx_km)


# ============================================================
# TN干扰距离条件分布 (公式4-5) — 距离单位: km
# ============================================================
def tn_interf_distance_cdf(r_n, r_0, r_TN, x_0):
    """
    TN干扰BS距离的条件CDF: F_{R_n|R_0}(r_n|r_0) (公式4)
    """
    if r_n <= r_0:
        return 0.0

    r_max = r_TN + x_0
    if r_n >= r_max:
        return 1.0

    f_r_n = single_point_cdf(r_n, r_TN, x_0)
    f_r_0 = single_point_cdf(r_0, r_TN, x_0)

    denom = 1.0 - f_r_0
    if denom < 1e-15:
        return 1.0

    return (f_r_n - f_r_0) / denom


def tn_interf_distance_pdf(r_n, r_0, r_TN, x_0, dx_km=1e-4):
    """
    TN干扰BS距离的条件PDF: f_{R_n|R_0}(r_n|r_0)
    """
    r_minus = max(r_0, r_n - dx_km)
    r_plus = r_n + dx_km
    cdf_plus = tn_interf_distance_cdf(r_plus, r_0, r_TN, x_0)
    cdf_minus = tn_interf_distance_cdf(r_minus, r_0, r_TN, x_0)
    denom = r_plus - r_minus
    if denom < 1e-15:
        return 0.0
    return (cdf_plus - cdf_minus) / denom


# ============================================================
# Laplace变换 — TN干扰 (公式9, Lemma 1)
# 使用CDF差分(Riemann-Stieltjes)替代数值PDF微分
# ============================================================
def tn_interference_laplace(s, r_0_km, r_TN_km, x_0_km, n_active,
                             p_tn, g_tn, alpha_tn, m_tn):
    """
    TN干扰的Laplace变换 (公式9)

    L_{I^TN}(s) = [integral_{r_0}^{r_TN+x_0} L_H(s*p*G*r^{-alpha}) * f_{R_n|R_0}(r_n|r_0) dr_n]^{n_active}

    使用CDF差分替代数值PDF，避免边界微分误差被n_active放大。
    对积分区间进行非均匀网格划分（log scale near r_0, linear elsewhere），
    在每个子区间上用中点处的L_H乘以CDF差值。

    参数:
        s: Laplace参数
        r_0_km: 服务距离 (km)
        r_TN_km: TN簇半径 (km)
        x_0_km: UE偏移 (km)
        n_active: 活跃干扰BS数 (N_c - 1)
        p_tn: TN BS发射功率 (W)
        g_tn: TN等效增益
        alpha_tn: TN路径损耗指数
        m_tn: TN Nakagami-m参数
    """
    r_max_km = r_TN_km + x_0_km

    if r_0_km >= r_max_km - 1e-6:
        return 1.0

    # 使用CDF差分(Riemann-Stieltjes)替代数值PDF
    # 划分N个子区间，用中点处的L_H乘以CDF差值
    N = 512
    # 在r_0附近L_H变化快，用更密的网格
    r_min_km = r_0_km

    # 创建非均匀网格：在r_0附近更密
    # 使用分段网格：靠近r_0用指数网格，远离r_0用均匀网格
    r_near = min(r_0_km + 2.0, r_max_km)  # 近区2km
    n_near = min(256, max(32, int(N * (r_near - r_min_km) / (r_max_km - r_min_km))))

    r_far = r_max_km
    n_far = N - n_near

    if n_far < 16:
        n_far = 16
        n_near = max(32, N - n_far)

    total = 0.0

    # 近区：密集网格
    if r_near > r_min_km + 1e-6:
        dr = (r_near - r_min_km) / n_near
        for i in range(n_near):
            r_left = r_min_km + i * dr
            r_right = r_min_km + (i + 1) * dr
            r_mid = (r_left + r_right) / 2

            r_mid_m = r_mid * KM2M
            l_h = nakagami_laplace(s, p_tn, g_tn, r_mid_m, alpha_tn, m_tn)

            cdf_right = tn_interf_distance_cdf(r_right, r_0_km, r_TN_km, x_0_km)
            cdf_left = tn_interf_distance_cdf(r_left, r_0_km, r_TN_km, x_0_km)

            total += l_h * (cdf_right - cdf_left)

    # 远区：均匀网格
    if r_far > r_near + 1e-6:
        dr = (r_far - r_near) / n_far
        for i in range(n_far):
            r_left = r_near + i * dr
            r_right = r_near + (i + 1) * dr
            r_mid = (r_left + r_right) / 2

            r_mid_m = r_mid * KM2M
            l_h = nakagami_laplace(s, p_tn, g_tn, r_mid_m, alpha_tn, m_tn)

            cdf_right = tn_interf_distance_cdf(r_right, r_0_km, r_TN_km, x_0_km)
            cdf_left = tn_interf_distance_cdf(r_left, r_0_km, r_TN_km, x_0_km)

            total += l_h * (cdf_right - cdf_left)

    total = max(0.0, min(total, 1.0))
    return total ** n_active


# ============================================================
# Laplace变换 — NTN下行干扰 (公式10, Lemma 2) — Case I
# ============================================================
def ntn_dl_laplace_zenith(s, altitude_km, p_ntn, g_ntn, alpha_ntn, m_ntn):
    """
    NTN下行干扰Laplace变换 — 卫星天顶简化版 (公式10简化)

    当卫星在UE天顶时, 干扰距离 = 卫星高度 a (m)
    L_{I^DL}(s) = (1 + s*p*g*a^{-alpha}/m)^{-m}

    参数:
        s: Laplace参数
        altitude_km: 卫星高度 (km)
        p_ntn: 卫星发射功率 (W)
        g_ntn: NTN下行等效增益
        alpha_ntn: NTN路径损耗指数
        m_ntn: NTN Nakagami-m参数
    """
    a_m = altitude_km * KM2M  # 转换为米
    return nakagami_laplace(s, p_ntn, g_ntn, a_m, alpha_ntn, m_ntn)


def ntn_dl_laplace_full(s, altitude_km, p_ntn, g_ntn, alpha_ntn, m_ntn):
    """
    NTN下行干扰Laplace变换 — 完整版 (公式10)

    L_{I^DL}(s) = integral_a^{r_max} L_H(s*p*G*r^{-alpha}) * f_{R_n}(r_n) dr_n
    其中 r_max = sqrt(2*R_earth*a + a^2), 单位km
    f_{R_n}(r_n) = 2*r_n / (r_max^2 - a^2)
    """
    r_max_km = np.sqrt(2 * R_EARTH * altitude_km + altitude_km ** 2)
    a_km = altitude_km

    def integrand(r_n_km):
        r_n_m = r_n_km * KM2M
        l_h = nakagami_laplace(s, p_ntn, g_ntn, r_n_m, alpha_ntn, m_ntn)
        pdf = 2 * r_n_km / (r_max_km ** 2 - a_km ** 2)
        return l_h * pdf

    try:
        val, _ = quad(integrand, a_km, r_max_km,
                      epsabs=QUAD_EPSABS, epsrel=QUAD_EPSREL, limit=QUAD_LIMIT)
    except Exception:
        return 1.0

    return val


# ============================================================
# 辅助函数: NTN UE距离分布 (Case II, 论文公式6)
# ============================================================
def circle_circle_intersection(d, r1, r2):
    """
    计算两个圆的交集面积
    参数:
        d: 两个圆心之间的距离
        r1, r2: 两个圆的半径
    """
    if d >= r1 + r2:
        return 0.0
    if d + r1 <= r2:
        return np.pi * r1 ** 2
    if d + r2 <= r1:
        return np.pi * r2 ** 2

    cos_a = (d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1)
    cos_b = (d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    cos_b = np.clip(cos_b, -1.0, 1.0)

    return r1 ** 2 * np.arccos(cos_a) + r2 ** 2 * np.arccos(cos_b) - 0.5 * np.sqrt(
        max(0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    )


def ntn_ue_distance_cdf(r_n, w_km, r_ntn_km, x_0_km):
    """
    NTN UE到TN UE的距离CDF (论文公式6)
    NTN UE均匀分布在以原点为圆心、内径w、外径r_ntn的环形区域
    TN UE位于(x_0, 0)

    参数:
        r_n: 距离 (km)
        w_km: 环形内径 = r_TN + r_iso
        r_ntn_km: 环形外径 = R_NTN (卫星波束覆盖半径 = 25 km)
        x_0_km: TN UE偏移 (km)
    """
    if r_n <= 0:
        return 0.0

    # 环形区域总面积
    total_area = np.pi * (r_ntn_km ** 2 - w_km ** 2)
    if total_area <= 0:
        return 0.0

    # 以(x_0, 0)为圆心、r_n为半径的圆与环形区域的交集面积
    # = 圆(x_0, r_n)与大圆(0, r_ntn)的交集 - 圆(x_0, r_n)与小圆(0, w)的交集
    area_outer = circle_circle_intersection(x_0_km, r_n, r_ntn_km)
    area_inner = circle_circle_intersection(x_0_km, r_n, w_km)
    area_intersection = max(0, area_outer - area_inner)

    return area_intersection / total_area


def ntn_ue_distance_pdf(r_n, w_km, r_ntn_km, x_0_km, dx=1e-4):
    """
    NTN UE距离PDF (数值差分)
    """
    cdf_plus = ntn_ue_distance_cdf(r_n + dx, w_km, r_ntn_km, x_0_km)
    cdf_minus = ntn_ue_distance_cdf(r_n - dx, w_km, r_ntn_km, x_0_km)
    return (cdf_plus - cdf_minus) / (2 * dx)


# ============================================================
# Laplace变换 — NTN上行干扰 (公式15, Lemma 3) — Case II
# 使用CDF差分(Riemann-Stieltjes)替代数值PDF微分
# ============================================================
def ntn_ul_laplace(s, r_TN_km, r_iso_km, x_0_km, n_u,
                   p_ntn_ul, g_ntn_ul, alpha_tn, m_ntn):
    """
    NTN上行干扰Laplace变换 (公式15)

    NTN UE均匀分布在环形区域:
    - 以原点为圆心
    - 内径 w = r_TN + r_iso
    - 外径 r_NTN = 25 km (卫星波束覆盖半径)

    TN UE位于(x_0, 0)

    使用CDF差分(Riemann-Stieltjes)替代数值PDF微分，
    避免边界处的微分误差被n_u放大（n_u可达2000）。

    参数:
        s: Laplace参数
        r_TN_km: TN簇半径 (km)
        r_iso_km: 隔离距离 (km)
        x_0_km: TN UE偏移 (km)
        n_u: NTN UE数量
        p_ntn_ul: NTN UE发射功率 (W)
        g_ntn_ul: NTN上行等效增益
        alpha_tn: 路径损耗指数
        m_ntn: NTN Nakagami-m参数
    """
    w_km = r_TN_km + r_iso_km  # 环形内径
    r_ntn_km = R_NTN            # 环形外径 = 25 km

    if w_km >= r_ntn_km:
        return 1.0

    # 距离范围：从最短到最长
    r_min_km = max(0, w_km - x_0_km)   # 最近NTN UE到TN UE的距离
    r_max_km = r_ntn_km + x_0_km       # 最远NTN UE到TN UE的距离

    if r_min_km >= r_max_km:
        return 1.0

    # 使用CDF差分(Riemann-Stieltjes)替代数值PDF
    # 划分N个子区间，用中点处的L_H乘以CDF差值
    N = 512
    total = 0.0
    dr = (r_max_km - r_min_km) / N

    for i in range(N):
        r_left = r_min_km + i * dr
        r_right = r_min_km + (i + 1) * dr
        r_mid = (r_left + r_right) / 2

        if r_mid <= 0:
            continue

        r_mid_m = r_mid * KM2M
        l_h = nakagami_laplace(s, p_ntn_ul, g_ntn_ul, r_mid_m, alpha_tn, m_ntn)

        cdf_right = ntn_ue_distance_cdf(r_right, w_km, r_ntn_km, x_0_km)
        cdf_left = ntn_ue_distance_cdf(r_left, w_km, r_ntn_km, x_0_km)

        total += l_h * (cdf_right - cdf_left)

    total = max(0.0, min(total, 1.0))
    return total ** n_u


# ============================================================
# 覆盖概率 — Case I (公式8, Theorem 1)
# ============================================================
def coverage_probability_case1(T_db, altitude_km, r_TN_km, x_0_km, load_factor,
                               p_tn=P_TN, g_tn=G_TN, p_ntn=P_NTN_DL, g_ntn=G_NTN_DL,
                               alpha_tn=ALPHA_TN, alpha_ntn=ALPHA_NTN,
                               m_tn=M_TN, m_ntn=M_NTN, sigma2=SIGMA2):
    """
    Case I覆盖概率: NTN下行干扰TN下行 (公式8)

    P_c(T) = integral_0^{r_TN+x_0} f_{R_0}(r_0) *
             [e^{-s*sigma^2} * L_{I^TN}(s) * L_{I^DL}(s)] dr_0

    其中 s = m_tn * T * r_0^{alpha_tn} / (p_tn * g_tn)
    注意: 这里的r_0在m空间计算s, 积分变量在km空间

    参数:
        T_db: SINR阈值 (dB)
        altitude_km: 卫星高度 (km)
        r_TN_km: TN簇半径 (km)
        x_0_km: UE偏移 (km)
        load_factor: TN负载因子 (0.25 or 1.0)
    返回:
        覆盖概率
    """
    T = 10 ** (T_db / 10)
    r_max_km = r_TN_km + x_0_km

    # 干扰BS数 = 总BS数 - 1 (服务BS)
    n_active = max(0, int(round(N_C * load_factor)) - 1)

    def integrand(r_0_km):
        if r_0_km <= 1e-4 or r_0_km >= r_max_km:
            return 0.0

        r_0_m = r_0_km * KM2M

        # s = m_tn * T * r_0^{alpha_tn} / (p_tn * g_tn)  [r_0 in meters]
        s = m_tn * T * r_0_m ** alpha_tn / (p_tn * g_tn)

        # 服务距离PDF (km空间)
        f_r0 = serving_distance_pdf(r_0_km, r_TN_km, x_0_km, N_C)

        # TN干扰Laplace变换
        l_tn = tn_interference_laplace(s, r_0_km, r_TN_km, x_0_km, n_active,
                                         p_tn, g_tn, alpha_tn, m_tn)

        # NTN下行干扰Laplace变换 (天顶简化)
        l_ntn = ntn_dl_laplace_zenith(s, altitude_km, p_ntn, g_ntn, alpha_ntn, m_ntn)

        # 噪声项
        noise_term = np.exp(-s * sigma2)

        return f_r0 * noise_term * l_tn * l_ntn

    try:
        val, _ = quad(integrand, 1e-3, r_max_km - 1e-3,
                      epsabs=1e-8, epsrel=1e-8, limit=QUAD_LIMIT)
    except Exception:
        val = 0.0

    return max(0.0, min(val, 1.0))


def coverage_probability_no_ntn(T_db, r_TN_km, x_0_km, load_factor,
                                 p_tn=P_TN, g_tn=G_TN,
                                 alpha_tn=ALPHA_TN, m_tn=M_TN, sigma2=SIGMA2):
    """
    无NTN干扰的基线覆盖概率
    """
    T = 10 ** (T_db / 10)
    r_max_km = r_TN_km + x_0_km

    n_active = max(0, int(round(N_C * load_factor)) - 1)

    def integrand(r_0_km):
        if r_0_km <= 1e-4 or r_0_km >= r_max_km:
            return 0.0

        r_0_m = r_0_km * KM2M
        s = m_tn * T * r_0_m ** alpha_tn / (p_tn * g_tn)
        f_r0 = serving_distance_pdf(r_0_km, r_TN_km, x_0_km, N_C)
        l_tn = tn_interference_laplace(s, r_0_km, r_TN_km, x_0_km, n_active,
                                         p_tn, g_tn, alpha_tn, m_tn)
        noise_term = np.exp(-s * sigma2)

        return f_r0 * noise_term * l_tn

    try:
        val, _ = quad(integrand, 1e-3, r_max_km - 1e-3,
                      epsabs=1e-8, epsrel=1e-8, limit=QUAD_LIMIT)
    except Exception:
        val = 0.0

    return max(0.0, min(val, 1.0))


# ============================================================
# 覆盖概率 — Case II (公式14, Theorem 3)
# ============================================================
def coverage_probability_case2(T_db, altitude_km, r_TN_km, r_iso_km, x_0_km,
                                n_u, load_factor,
                                p_tn=P_TN, g_tn=G_TN,
                                p_ntn_ul=P_NTN_UL, g_ntn_ul=G_NTN_UL,
                                alpha_tn=ALPHA_TN, m_tn=M_TN, m_ntn=M_NTN,
                                sigma2=SIGMA2):
    """
    Case II覆盖概率: NTN上行干扰TN下行 (公式14)
    """
    T = 10 ** (T_db / 10)
    r_max_km = r_TN_km + x_0_km

    n_active = max(0, int(round(N_C * load_factor)) - 1)

    def integrand(r_0_km):
        if r_0_km <= 1e-4 or r_0_km >= r_max_km:
            return 0.0

        r_0_m = r_0_km * KM2M
        s = m_tn * T * r_0_m ** alpha_tn / (p_tn * g_tn)
        f_r0 = serving_distance_pdf(r_0_km, r_TN_km, x_0_km, N_C)
        l_tn = tn_interference_laplace(s, r_0_km, r_TN_km, x_0_km, n_active,
                                         p_tn, g_tn, alpha_tn, m_tn)
        l_ntn_ul = ntn_ul_laplace(s, r_TN_km, r_iso_km, x_0_km, n_u,
                                   p_ntn_ul, g_ntn_ul, ALPHA_NTN, m_ntn)
        noise_term = np.exp(-s * sigma2)

        return f_r0 * noise_term * l_tn * l_ntn_ul

    try:
        val, _ = quad(integrand, 1e-3, r_max_km - 1e-3,
                      epsabs=1e-8, epsrel=1e-8, limit=QUAD_LIMIT)
    except Exception:
        val = 0.0

    return max(0.0, min(val, 1.0))


# ============================================================
# Monte Carlo仿真验证
# ============================================================
def mc_coverage_case1(T_db, altitude_km, r_TN_km, x_0_km, load_factor,
                      n_trials=50000,
                      p_tn=P_TN, g_tn=G_TN, p_ntn=P_NTN_DL, g_ntn=G_NTN_DL,
                      alpha_tn=ALPHA_TN, alpha_ntn=ALPHA_NTN,
                      m_tn=M_TN, m_ntn=M_NTN, sigma2=SIGMA2):
    """
    Monte Carlo仿真验证 Case I
    所有距离在路径损耗中以m为单位
    """
    T = 10 ** (T_db / 10)
    n_active = max(1, int(round(N_C * load_factor)))

    count_covered = 0
    for _ in range(n_trials):
        # 生成BS位置 (km空间, 然后转m)
        bs_positions = []
        while len(bs_positions) < n_active:
            x = np.random.uniform(-r_TN_km, r_TN_km)
            y = np.random.uniform(-r_TN_km, r_TN_km)
            if x ** 2 + y ** 2 <= r_TN_km ** 2:
                bs_positions.append((x + x_0_km, y))

        ue_pos = (0.0, 0.0)

        # 计算距离 (km), 然后转m
        distances_km = [np.sqrt((bx - ue_pos[0]) ** 2 + (by - ue_pos[1]) ** 2)
                        for bx, by in bs_positions]
        distances_m = [d * KM2M for d in distances_km]

        sorted_dist_m = sorted(distances_m)
        r_0_m = sorted_dist_m[0]

        # 服务信号
        h_0 = np.random.gamma(m_tn, 1.0 / m_tn)
        signal = p_tn * g_tn * h_0 * r_0_m ** (-alpha_tn)

        # TN干扰
        interference_tn = 0.0
        for d_m in sorted_dist_m[1:]:
            h = np.random.gamma(m_tn, 1.0 / m_tn)
            interference_tn += p_tn * g_tn * h * d_m ** (-alpha_tn)

        # NTN下行干扰 (天顶, 距离=高度m)
        altitude_m = altitude_km * KM2M
        h_ntn = np.random.gamma(m_ntn, 1.0 / m_ntn)
        interference_ntn = p_ntn * g_ntn * h_ntn * altitude_m ** (-alpha_ntn)

        sinr = signal / (interference_tn + interference_ntn + sigma2)

        if sinr > T:
            count_covered += 1

    return count_covered / n_trials
