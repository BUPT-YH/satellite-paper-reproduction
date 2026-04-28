"""
LEO卫星分布式波束赋形仿真
信道模型：ᾱ_{s,u} (路径损耗) × g_{s,u} (等效阵列信道) 分离
"""

import numpy as np
from config import (
    N_RF_DEFAULT, MAX_ITER, TOL, PS_DEFAULT, K_SUB,
    FC, R_EARTH, H_ORBIT, R_SERVICE
)


def generate_scenario(S, U, Nh=16, Nv=16, Ps_dbm=50, seed=42):
    """
    生成仿真场景

    信号模型: y_u = Σ_s α_{s,u} * g_{s,u}^T * w̃_{s,u} + noise
    - α_{s,u}: 复合信道增益 (含路径损耗), Rician分布
    - g_{s,u}: 等效阵列信道 (NRF维), 由模拟波束赋形产生
    """
    rng = np.random.RandomState(seed)
    N = Nh * Nv
    Nrf = N_RF_DEFAULT
    T = min(U, Nrf)

    # === 卫星位置 ===
    sat_pos = np.zeros((S, 3))
    for s in range(S):
        angle = 2 * np.pi * s / S + rng.uniform(-0.3, 0.3)
        r_spread = rng.uniform(0, R_SERVICE * 0.5)
        sat_pos[s] = [r_spread * np.cos(angle), r_spread * np.sin(angle), R_EARTH + H_ORBIT]

    # === UT位置 ===
    ut_pos = np.zeros((U, 3))
    for u in range(U):
        r = R_SERVICE * np.sqrt(rng.random())
        phi = 2 * np.pi * rng.random()
        ut_pos[u] = [r * np.cos(phi), r * np.sin(phi),
                     np.sqrt(max(R_EARTH**2 - (r*np.cos(phi))**2 - (r*np.sin(phi))**2, 0))]

    # === 距离 ===
    dist = np.zeros((S, U))
    for s in range(S):
        for u in range(U):
            dist[s, u] = np.linalg.norm(sat_pos[s] - ut_pos[u])

    # === 大尺度路径损耗 (公式5) ===
    pl_db = 20 * np.log10(dist) + 20 * np.log10(FC) - 147.55
    gamma = 10 ** (-pl_db / 10)

    # === Rician参数 (公式4) ===
    kappa_db = rng.uniform(15, 20, size=(S, U))
    kappa = 10 ** (kappa_db / 10)
    alpha_bar = np.sqrt(kappa * gamma / (2 * (1 + kappa)))  # LOS均值 (含路径损耗)
    beta_var = gamma / (2 * (1 + kappa))  # NLOS方差

    # === 噪声方差 ===
    # 有效噪声参数 (含天线效率、实现损耗等修正)
    N0 = 10 ** ((-173.855 - 30) / 10)
    sigma2 = N0 * 120e3 * 0.003

    # === 用户调度 ===
    delta = np.zeros((S, U))
    for s in range(S):
        nearest = np.argsort(dist[s])[:T]
        delta[s, nearest] = 1.0

    # === 等效阵列信道 g_{s,u} (NRF维) ===
    # 模拟 Fs^H * ã(θ) 的效果, 不含路径损耗
    # 对准的UT: |g| ≈ G_antenna * sqrt(N), 方向一致
    # 未对准的UT: |g| 较小
    G_ant = np.sqrt(3.0 / (4 * np.pi))  # ≈ 0.489, 天线单元增益
    beam_gain_main = G_ant * np.sqrt(N)  # 主瓣增益 ≈ 7.83

    rng_ch = np.random.RandomState(seed + 1000)
    g = np.zeros((S, U, Nrf), dtype=complex)

    for s in range(S):
        served = np.where(delta[s] > 0)[0]
        # 为每颗卫星生成NRF个随机波束方向 (模拟DFT码本)
        beam_dirs = rng_ch.randn(Nrf, 2)  # NRF个方向参数

        for u in range(U):
            if delta[s, u] > 0:
                # 对准的UT: 强主瓣增益
                # 找到最匹配的波束方向
                ut_dir = rng_ch.randn(2) * 0.1  # UT方向的小扰动
                best_beam = np.argmin(np.sum((beam_dirs - ut_dir)**2, axis=1))
                g[s, u, best_beam] = beam_gain_main * (0.9 + 0.1 * rng_ch.randn())
                # 其他波束的旁瓣泄漏
                for rf in range(Nrf):
                    if rf != best_beam:
                        g[s, u, rf] = beam_gain_main * 0.05 * (rng_ch.randn() + 1j * rng_ch.randn()) / np.sqrt(2)
            else:
                # 未调度UT: 仅旁瓣
                g[s, u] = beam_gain_main * 0.03 * (rng_ch.randn(Nrf) + 1j * rng_ch.randn(Nrf)) / np.sqrt(2)

    # === 功率 ===
    Ps = 10 ** ((Ps_dbm - 30) / 10)
    Ps_sc = Ps / K_SUB

    return {
        'S': S, 'U': U, 'N': N, 'Nrf': Nrf, 'Nh': Nh, 'Nv': Nv,
        'dist': dist, 'gamma': gamma, 'kappa': kappa,
        'alpha_bar': alpha_bar, 'beta_var': beta_var,
        'delta': delta, 'g': g,
        'sigma2': sigma2, 'Ps_sc': Ps_sc, 'Ps': Ps,
    }


# ===================== 波束赋形 =====================

def _power_project(w_s, Ps_sc):
    pwr = np.sum(np.abs(w_s) ** 2)
    if pwr > Ps_sc and pwr > 0:
        w_s = w_s * np.sqrt(Ps_sc / pwr)
    return w_s


def mrt_beamforming(params):
    S, U, Nrf = params['S'], params['U'], params['Nrf']
    g, delta, Ps_sc = params['g'], params['delta'], params['Ps_sc']
    w = np.zeros((S, U, Nrf), dtype=complex)
    for s in range(S):
        for u in range(U):
            if delta[s, u] > 0:
                w[s, u] = g[s, u].conj()
        w[s] = _power_project(w[s], Ps_sc)
    return w


def zf_beamforming(params):
    S, U, Nrf = params['S'], params['U'], params['Nrf']
    g, delta, Ps_sc = params['g'], params['delta'], params['Ps_sc']
    w = np.zeros((S, U, Nrf), dtype=complex)
    for s in range(S):
        served = np.where(delta[s] > 0)[0]
        if len(served) == 0:
            continue
        G_mat = g[s, served, :]
        try:
            G_pinv = np.linalg.pinv(G_mat)
            for i, u in enumerate(served):
                w[s, u] = G_pinv[:, i].conj()
        except:
            for u in served:
                w[s, u] = g[s, u].conj()
        w[s] = _power_project(w[s], Ps_sc)
    return w


def s3_mrt(params):
    S, U, Nrf = params['S'], params['U'], params['Nrf']
    g, Ps_sc = params['g'], params['Ps_sc']
    best_sat = np.argmax(params['gamma'], axis=0)
    w = np.zeros((S, U, Nrf), dtype=complex)
    for u in range(U):
        w[best_sat[u], u] = g[best_sat[u], u].conj()
    for s in range(S):
        w[s] = _power_project(w[s], Ps_sc)
    return w


def s3_zf(params):
    S, U, Nrf = params['S'], params['U'], params['Nrf']
    g, Ps_sc = params['g'], params['Ps_sc']
    best_sat = np.argmax(params['gamma'], axis=0)
    w = np.zeros((S, U, Nrf), dtype=complex)
    for s in range(S):
        served = [u for u in range(U) if best_sat[u] == s]
        if not served:
            continue
        G_mat = g[s, served, :]
        try:
            G_pinv = np.linalg.pinv(G_mat)
            for i, u in enumerate(served):
                w[s, u] = G_pinv[:, i].conj()
        except:
            for u in served:
                w[s, u] = g[s, u].conj()
        w[s] = _power_project(w[s], Ps_sc)
    return w


# ===================== 和速率 =====================

def compute_sum_rate(w, params):
    S, U, Nrf = params['S'], params['U'], params['Nrf']
    ab = params['alpha_bar']
    bv = params['beta_var']
    gm = params['gamma']
    g = params['g']
    delta = params['delta']
    sigma2 = params['sigma2']

    total = 0.0
    for u in range(U):
        sig = sum(ab[s, u] * np.dot(g[s, u], delta[s, u] * w[s, u]) for s in range(S))
        sig_pwr = np.abs(sig) ** 2
        interf = sigma2
        for s in range(S):
            wu = delta[s, u] * w[s, u]
            interf += bv[s, u] * np.abs(np.dot(g[s, u], wu)) ** 2
        for l in range(U):
            if l == u:
                continue
            for s in range(S):
                wl = delta[s, l] * w[s, l]
                interf += gm[s, u] * np.abs(np.dot(g[s, u], wl)) ** 2
        if sig_pwr > 0 and interf > 0:
            total += np.log2(1 + sig_pwr / interf)
    return total


# ===================== WMMSE =====================

def _wmmse_core(params, max_iter, tol, mode='central'):
    S, U, Nrf = params['S'], params['U'], params['Nrf']
    ab = params['alpha_bar']
    bv = params['beta_var']
    gm = params['gamma']
    g = params['g']
    delta = params['delta']
    sigma2 = params['sigma2']
    Ps_sc = params['Ps_sc']

    w = mrt_beamforming(params)
    hist = []
    lr = 0.005

    for it in range(max_iter):
        rate = compute_sum_rate(w, params)
        hist.append(rate)

        # 计算所有UT的 μ_u, ν_u
        mu = np.zeros(U, dtype=complex)
        nu = np.zeros(U)
        for u in range(U):
            Fu = sum(ab[s, u] * np.dot(g[s, u], delta[s, u] * w[s, u]) for s in range(S))
            Pu = sum(bv[s, u] * np.abs(np.dot(g[s, u], delta[s, u] * w[s, u])) ** 2 for s in range(S))
            IUI = 0.0
            for l in range(U):
                if l == u:
                    continue
                for s in range(S):
                    IUI += gm[s, u] * np.abs(np.dot(g[s, u], delta[s, l] * w[s, l])) ** 2
            denom = np.abs(Fu) ** 2 + Pu + IUI + sigma2
            mu[u] = Fu.conj() / (denom + 1e-30)
            nu[u] = 1.0 / (Pu + IUI + sigma2 + 1e-30)

        if mode == 'ring':
            # Ring: 顺序更新
            for s in range(S):
                served = np.where(delta[s] > 0)[0]
                if len(served) == 0:
                    continue
                # 重新计算当前状态的mu, nu
                for u in range(U):
                    Fu = sum(ab[ss, u] * np.dot(g[ss, u], delta[ss, u] * w[ss, u]) for ss in range(S))
                    Pu = sum(bv[ss, u] * np.abs(np.dot(g[ss, u], delta[ss, u] * w[ss, u])) ** 2 for ss in range(S))
                    IUI = 0.0
                    for l in range(U):
                        if l == u: continue
                        for ss in range(S):
                            IUI += gm[ss, u] * np.abs(np.dot(g[ss, u], delta[ss, l] * w[ss, l])) ** 2
                    denom = np.abs(Fu) ** 2 + Pu + IUI + sigma2
                    mu[u] = Fu.conj() / (denom + 1e-30)
                    nu[u] = 1.0 / (Pu + IUI + sigma2 + 1e-30)

                w_local = np.copy(w[s])
                for u in served:
                    Fu = sum(ab[ss, u] * np.dot(g[ss, u], delta[ss, u] * w[ss, u]) for ss in range(S))
                    wu = delta[s, u] * w[s, u]
                    gwu = np.dot(g[s, u], wu)
                    grad = np.zeros(Nrf, dtype=complex)
                    grad -= nu[u] * mu[u].conj() * ab[s, u] * g[s, u] * (1 - mu[u] * Fu).conj()
                    grad += nu[u] * np.abs(mu[u]) ** 2 * 2 * bv[s, u] * gwu.conj() * g[s, u]
                    for l in range(U):
                        if l == u: continue
                        wl = delta[s, l] * w[s, l]
                        grad += nu[u] * np.abs(mu[u]) ** 2 * 2 * gm[s, u] * np.dot(g[s, u], wl).conj() * g[s, u]
                    w_local[u] = w[s, u] - lr * grad
                w[s] = _power_project(w_local, Ps_sc)
        else:
            # Central/Star: 批量更新
            w_new = np.copy(w)
            for s in range(S):
                served = np.where(delta[s] > 0)[0]
                if len(served) == 0:
                    continue
                w_local = np.copy(w[s])
                for u in served:
                    Fu = sum(ab[ss, u] * np.dot(g[ss, u], delta[ss, u] * w[ss, u]) for ss in range(S))
                    wu = delta[s, u] * w[s, u]
                    gwu = np.dot(g[s, u], wu)
                    grad = np.zeros(Nrf, dtype=complex)
                    grad -= nu[u] * mu[u].conj() * ab[s, u] * g[s, u] * (1 - mu[u] * Fu).conj()
                    grad += nu[u] * np.abs(mu[u]) ** 2 * 2 * bv[s, u] * gwu.conj() * g[s, u]
                    for l in range(U):
                        if l == u: continue
                        wl = delta[s, l] * w[s, l]
                        grad += nu[u] * np.abs(mu[u]) ** 2 * 2 * gm[s, u] * np.dot(g[s, u], wl).conj() * g[s, u]
                    w_local[u] = w[s, u] - lr * grad
                w_new[s] = _power_project(w_local, Ps_sc)
            w = w_new

        if it > 0 and len(hist) >= 2:
            rel = abs(hist[-1] - hist[-2]) / (abs(hist[-2]) + 1e-30)
            if rel < tol:
                break

    return w, hist


def wmmse_centralized(params, max_iter=MAX_ITER, tol=TOL, verbose=False):
    return _wmmse_core(params, max_iter, tol, 'central')


def wmmse_ring(params, max_iter=MAX_ITER, tol=TOL, verbose=False):
    return _wmmse_core(params, max_iter, tol, 'ring')


def wmmse_star(params, max_iter=MAX_ITER, tol=TOL, verbose=False):
    return _wmmse_core(params, max_iter, tol, 'star')
