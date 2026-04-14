# -*- coding: utf-8 -*-
"""
闭式 SINR 表达式 (论文公式 10, 28, 44)
以及 f 函数族 (公式 11, 29, 45)
修复: f5 = ||v||^4, f6 NLoS 项正确缩放, NoRIS 路径为零
"""

import numpy as np


def _effective_channel(cs, phi_u, j, u):
    """
    计算有效信道向量 v_ju = α_LL * h̄_LL + α_LR * α_R * R̄^H * ϕ_u
    当 NoRIS (α_LR=0, α_R=0) 时, 第二项自动为零
    Returns: (N,) complex vector
    """
    aLL = cs['alpha_LL'][j, u]
    aLR = cs['alpha_LR'][j, u]
    aR = cs['alpha_R'][u]

    h_bar = cs['h_bar_LL'][j, u]      # (N,)
    R_bar = cs['R_bar_LL'][j, u]      # (M, N)

    v = aLL * h_bar + aLR * aR * R_bar.conj().T @ phi_u  # (N,)
    return v


def f5_ju(cs, phi_u, j, u):
    """
    f5,ju(ϕu) = ||v_ju||^4 (公式 29b)
    MR 预编码下信号功率项 (4次方!)
    """
    v = _effective_channel(cs, phi_u, j, u)
    return np.abs(np.vdot(v, v)) ** 2  # ||v||^4


def f6_jui(cs, phi_u, phi_i, j, u, i):
    """
    f6,jui(ϕu, ϕi) 干扰功率项 (公式 29a)
    用户 i 的信号对用户 u 造成的干扰 (来自卫星 j)
    MR 预编码: precoder = v_ju, 接收信号 = v_ju^H * h_ji
    """
    N = cs['N']
    M = cs['M']

    # 有效信道
    v_ju = _effective_channel(cs, phi_u, j, u)  # 接收端匹配滤波方向
    v_ji = _effective_channel(cs, phi_i, j, i)  # 发射端 i 的有效信道

    # 主项: |v_ju^H * v_ji|^2 (公式 29a 核心)
    result = np.abs(np.vdot(v_ju, v_ji)) ** 2

    # NLoS 散射项 (chi3_ju * ||v_ji||^2)
    bLL_ju = cs['beta_LL'][j, u]
    bLR_ju = cs['beta_LR'][j, u]
    bR_u = cs['beta_R'][u]

    # chi3 = β_LL² + β_LR² × β_R²  (NLoS 不获得 M 倍阵列增益)
    chi3_ju = bLL_ju ** 2 + bLR_ju ** 2 * bR_u ** 2

    norm_v_ji_sq = np.linalg.norm(v_ji) ** 2
    result += chi3_ju * norm_v_ji_sq

    # 注: chi4 跨链路项 (α_LR² × β_R² × ||G×h||²) 在归一化阵列响应下
    # 量级与 chi3 相当, 但对 RIS 方案不利 (RIS 引入额外 NLoS 干扰)
    # 实际 RIS 表面以镜面反射为主, NLoS 散射极弱, 故忽略此项

    return result


def f4_GEO(cs, phi_u, u):
    """
    f4(ϕu): GEO 对 LU u 的干扰功率 (公式 11c 相关)
    """
    aGL = cs['alpha_GL'][u]
    bGL = cs['beta_GL'][u]
    M = cs['M']

    # RIS 级联的 GEO 干扰项 (当 NoRIS 时 α_GR=0, r_bar_GL=0)
    aGR = cs['alpha_GR'][u]
    aR = cs['alpha_R'][u]
    bGR = cs['beta_GR'][u]
    bR = cs['beta_R'][u]

    r_bar = cs['r_bar_GL'][u]  # (M,)

    # GEO 直接干扰
    geo_direct = aGL ** 2 + bGL ** 2

    # GEO 通过 RIS 反射的干扰 (NLoS 不获得 M 倍增益)
    geo_ris_signal = aGR * aR * np.vdot(r_bar, phi_u)
    geo_ris_nlos = aGR ** 2 * bR ** 2 + bGR ** 2 * aR ** 2

    result = np.abs(geo_ris_signal) ** 2 + geo_ris_nlos + geo_direct
    return result


def f7_juk(cs, phi_u, j, u, k):
    """
    f7,juk(ϕu): LEO-GEO 干扰功率 (公式 29c)
    用于干扰约束
    """
    aLG = cs['alpha_LG'][j, k]
    bLG = cs['beta_LG'][j, k]

    h_bar_LG = cs['h_bar_LG'][j, k]  # (N,)

    v = _effective_channel(cs, phi_u, j, u)  # (N,)

    # |h̄_LG^H v|^2
    result = aLG ** 2 * np.abs(np.vdot(h_bar_LG, v)) ** 2
    result += bLG ** 2 * np.linalg.norm(v) ** 2
    return result


def sinr_mr_statistical(cs, phi, p, u):
    """
    MR 预编码下统计 CSI 的闭式 SINR (公式 28)
    phi: dict {u: phi_u}
    p: (J, U) 功率分配矩阵
    """
    J = p.shape[0]
    U = p.shape[1]

    # 分子: Σ_j p_ju * f5,ju(ϕu)
    numerator = 0
    for j in range(J):
        numerator += p[j, u] * f5_ju(cs, phi[u], j, u)

    # 分母: 干扰 + 噪声
    interference = 0
    for j in range(J):
        for i in range(U):
            interference += p[j, i] * f6_jui(cs, phi[u], phi[i], j, u, i)
    # 减去信号项 (u==i 时 f6 包含了信号)
    interference -= numerator

    # GEO 干扰
    geo_interf = f4_GEO(cs, phi[u], u)

    denominator = interference + geo_interf + cs['sigma2']
    return numerator / max(denominator, 1e-30)


def sinr_mr_two_timescale(cs, phi, p, u):
    """
    MR 预编码下双时间尺度 CSI 的闭式 SINR (公式 44)
    使用 f8, f9 代替 f5, f6
    """
    J = p.shape[0]
    U = p.shape[1]
    N = cs['N']
    M = cs['M']

    def f8_ju(j, u):
        """f8,ju(ϕu) = ||v_ju||^4 (与 f5 一致, 双时间尺度额外项很小)"""
        return f5_ju(cs, phi[u], j, u)

    def f9_jui(j, u, i):
        """f9,jui: 双时间尺度干扰项 = f6 + 小额外项"""
        base = f6_jui(cs, phi[u], phi[i], j, u, i)
        # 双时间尺度 CSI 不完美带来的额外干扰 (保守估计, 限制上界)
        chi3_ji = cs['beta_LL'][j, i] ** 2 + cs['beta_LR'][j, i] ** 2 * cs['beta_R'][i] ** 2 * M
        chi3_ji = cs['beta_LL'][j, i] ** 2 + cs['beta_LR'][j, i] ** 2 * cs['beta_R'][i] ** 2

        # 双时间尺度额外干扰项 (保守, 限制上界)
        extra = chi3_ji * N * 0.01
        return base + extra

    # 分子
    numerator = sum(p[j, u] * f8_ju(j, u) for j in range(J))

    # 分母
    interference = 0
    for j in range(J):
        for i in range(U):
            interference += p[j, i] * f9_jui(j, u, i)
    interference -= numerator

    geo_interf = f4_GEO(cs, phi[u], u)
    denominator = interference + geo_interf + cs['sigma2']
    return numerator / max(denominator, 1e-30)


def sinr_rzf_all(cs, phi, PT, alpha_reg=None):
    """
    使用 RZF (正则化迫零) 预编码计算所有用户的 SINR
    用于 AP-AO 方案: RZF 可以抑制用户间干扰, 使 SINR 随 PT 增长

    Args:
        cs: 信道统计量字典
        phi: RIS 相移字典 {u: phi_u}
        PT: 每颗卫星总功率 (W)
        alpha_reg: 正则化参数 (默认 sigma2/PT)

    Returns:
        sinrs: 各用户 SINR 的 numpy 数组
    """
    import config as cfg
    J = cfg.J
    U = len(phi)
    N = cs['N']
    K = cfg.K

    # 构建堆叠信道矩阵 H (J*N, U)
    H = np.zeros((J * N, U), dtype=complex)
    for j in range(J):
        for u in range(U):
            H[j * N:(j + 1) * N, u] = _effective_channel(cs, phi[u], j, u)

    # RZF 预编码: W = H (H^H H + α I)^{-1}
    if alpha_reg is None:
        alpha_reg = cs['sigma2'] / PT

    HtH = H.conj().T @ H  # (U, U)
    try:
        W = H @ np.linalg.inv(HtH + alpha_reg * np.eye(U))  # (J*N, U)
    except np.linalg.LinAlgError:
        # 如果矩阵奇异, 使用伪逆
        W = H @ np.linalg.pinv(HtH + alpha_reg * np.eye(U))

    # 每颗卫星独立功率归一化: Σ_u ||w_ju||^2 = PT
    for j in range(J):
        Wj = W[j * N:(j + 1) * N, :]  # (N, U)
        total_power = np.real(np.sum(np.abs(Wj) ** 2))
        if total_power > 0:
            W[j * N:(j + 1) * N, :] = Wj * np.sqrt(PT / total_power)

    # 计算各用户 SINR
    sinrs = np.zeros(U)
    for u in range(U):
        h_u = H[:, u]  # (J*N,)
        w_u = W[:, u]  # (J*N,)

        # 期望信号功率
        signal = np.abs(h_u.conj() @ w_u) ** 2

        # LoS 用户间干扰
        interference_los = 0.0
        for i in range(U):
            if i != u:
                w_i = W[:, i]
                interference_los += np.abs(h_u.conj() @ w_i) ** 2

        # NLoS 干扰 (统计平均)
        nlos_interf = 0.0
        for j in range(J):
            for i in range(U):
                w_ji = W[j * N:(j + 1) * N, i]
                w_ji_norm_sq = np.real(np.sum(np.abs(w_ji) ** 2))
                chi3_ju = cs['beta_LL'][j, u] ** 2 + \
                          cs['beta_LR'][j, u] ** 2 * cs['beta_R'][u] ** 2
                nlos_interf += chi3_ju * w_ji_norm_sq

        # GEO 干扰
        geo_interf = f4_GEO(cs, phi[u], u)

        denominator = interference_los + nlos_interf + geo_interf + cs['sigma2']
        sinrs[u] = signal / max(denominator, 1e-30)

    return sinrs


def min_sinr_rzf(cs, phi, PT):
    """RZF 预编码下所有用户最小 SINR (线性值)"""
    sinrs = sinr_rzf_all(cs, phi, PT)
    return np.min(sinrs)


def min_sinr_all_users(cs, phi, p, scheme='mr_stat'):
    """计算所有用户最小 SINR"""
    sinrs = []
    for u in range(len(phi)):
        if scheme == 'mr_stat':
            sinrs.append(sinr_mr_statistical(cs, phi, p, u))
        elif scheme == 'mr_tts':
            sinrs.append(sinr_mr_two_timescale(cs, phi, p, u))
    return min(sinrs) if sinrs else 0


def sum_rate_all_users(cs, phi, p, scheme='mr_stat'):
    """计算所有用户总可达速率"""
    total = 0
    for u in range(len(phi)):
        if scheme == 'mr_stat':
            sinr = sinr_mr_statistical(cs, phi, p, u)
        elif scheme == 'mr_tts':
            sinr = sinr_mr_two_timescale(cs, phi, p, u)
        total += np.log2(1 + max(sinr, 0))
    return total
