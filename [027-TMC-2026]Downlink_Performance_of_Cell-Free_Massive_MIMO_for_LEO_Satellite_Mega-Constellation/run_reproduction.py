"""
一键复现脚本 - 全部10张数据图
论文: Downlink Performance of Cell-Free Massive MIMO for LEO Satellite Mega-Constellation
期刊: IEEE TMC, 2026
"""

import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from config import get_params, compute_distance_bounds, compute_avg_ut_number
from simulation import (
    mc_coverage, mc_dss_ccdf, mc_sinr_single, mc_sinr_single_cell_based,
    nakagami_sq, generate_sap_positions, generate_starlink_constellation,
    avg_mui, avg_isi, coverage_analytical,
)
from plotting import *

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT, exist_ok=True)


def extract_concept_figures():
    """从PDF提取场景图"""
    import fitz
    from PIL import Image
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '[027-TMC-2026]Downlink_Performance_of_Cell-Free_Massive_MIMO_for_LEO_Satellite_Mega-Constellation.pdf')
    doc = fitz.open(pdf_path)
    scale = 300 / 72
    specs = {
        # Fig.1: 跨栏宽图 (右栏无独立文字)
        2: [('fig1_constellations', 55, 42, 545, 219)],
        # Fig.2: 左栏图 (右栏有公式文字 Eq.11/12)
        5: [('fig2_stochastic_geometry', 42, 42, 293, 218)],
        # Fig.3: 左栏图 (右栏有公式文字 Eq.23)
        7: [('fig3_distance_statistics', 42, 42, 293, 212)],
    }
    for pg, fig_specs in specs.items():
        page = doc[pg]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        for name, x0, y0, x1, y1 in fig_specs:
            cropped = img.crop((int(x0*scale), int(y0*scale), int(x1*scale), int(y1*scale)))
            cropped.save(f'{OUTPUT}/{name}.png', dpi=(300, 300))
            print(f'  Extracted {name}.png')
    doc.close()


def run_fig4():
    """Fig.4: PPP vs Starlink (含干扰的完整SINR)"""
    print('\n=== Fig.4: PPP vs Starlink ===')
    p = get_params(HS=500)
    gamma_th = np.arange(-5, 16, 1, dtype=float)
    RE, RS = p['RE'], p['RS']
    rS_min, rS_max, rmax = compute_distance_bounds(p)
    phi_U = compute_avg_ut_number(p)

    # PPP (MC仿真)
    ppp_cov = mc_coverage(p, gamma_th, n_realizations=5000, seed=42)
    print('  PPP done')

    # Starlink (含MUI和ISI)
    inclinations = [33, 43, 53]
    planes = 28
    total_sats = int(p['lambda_S'] * 4 * np.pi * RS**2)
    sats_per_plane = total_sats // (3 * planes)

    rand_covs, fixed_covs = [], []
    for lat_deg in [20, 40, 60]:
        lat = np.deg2rad(lat_deg)
        ut_pos = np.array([RE*np.cos(lat), 0, RE*np.sin(lat)])
        for random_state in [True, False]:
            rng = np.random.default_rng(42)
            sinr_arr = np.zeros(3000)
            for i in range(3000):
                sap = generate_starlink_constellation(RS, inclinations, planes, sats_per_plane, rng, random_state)
                dists = np.linalg.norm(sap - ut_pos, axis=1)
                s_mask = (dists >= rS_min) & (dists <= rS_max)
                i_mask = (dists > rS_max) & (dists <= rmax)
                N_ser = np.sum(s_mask)
                N_int = np.sum(i_mask)
                if N_ser == 0:
                    sinr_arr[i] = -1e10; continue

                # 期望信号 (coherent combining)
                d_ser = dists[s_mask]; beta_ser = d_ser**(-2)
                h2_ser = nakagami_sq(p['m'], N_ser, rng)
                DS = np.sum(np.sqrt(p['rho_d']/phi_U*p['Gml'])*np.sqrt(beta_ser)*np.sqrt(h2_ser))
                sig_power = DS**2

                # MUI (多用户干扰)
                N_other = max(0, int(round(phi_U)) - 1)
                mui_power = 0.0
                if N_other > 0:
                    h2_mui = nakagami_sq(p['m'], (N_ser, N_other), rng)
                    mui_power = np.sum(p['rho_d']/phi_U * p['Gml'] * beta_ser[:,None] * h2_mui)

                # ISI (星间干扰, 旁瓣)
                isi_power = 0.0
                if N_int > 0:
                    d_int = dists[i_mask]; beta_int = d_int**(-2)
                    h2_int = nakagami_sq(p['m'], N_int, rng)
                    isi_power = np.sum(p['rho_d'] * p['Gsl'] * beta_int * h2_int)

                sinr_arr[i] = sig_power / (mui_power + isi_power + p['sigma2'])

            cov = np.array([np.mean(sinr_arr > 10**(g/10)) for g in gamma_th])
            if random_state:
                rand_covs.append(cov)
            else:
                fixed_covs.append(cov)
        print(f'  Starlink lat={lat_deg}° done')

    labels = ['PPP Model'] + [f'Starlink(R) {l}°' for l in [20,40,60]] + \
             [f'Starlink(F) {l}°' for l in [20,40,60]]
    plot_fig4(gamma_th, ppp_cov, rand_covs, fixed_covs, labels, OUTPUT)


def run_fig5():
    """Fig.5: DSS CCDF (η=90°: 所有可见SAP均为服务SAP, 无ISI)"""
    print('\n=== Fig.5: DSS CCDF ===')
    p = get_params(eta=np.deg2rad(90))
    # DSS实际值在~5e-6到~1.5e-5范围，调整x_range
    x_range = np.linspace(4e-6, 1.6e-5, 60)

    # Perfect CSI: 大样本MC
    ccdf_perf, _ = mc_dss_ccdf(p, x_range, n_realizations=20000, seed=42)
    print('  Perfect CSI done')

    ccdf_imp = {}
    for tp in [20, 100, 200]:
        ccdf_imp[tp], _ = mc_dss_ccdf(p, x_range, n_realizations=10000, seed=42,
                                        use_lmmse=True, tau_p_val=tp)
        print(f'  tau_p={tp} done')

    plot_fig5(x_range, ccdf_perf, ccdf_imp, OUTPUT)


def run_fig6():
    """Fig.6: 不同Nakagami m参数 (Analytical + MC Simulation)
    使用Theorem 1框架: S_hat分布(MC) + 确定性平均干扰阈值
    这样m的敏感性通过S_hat分布直接体现, 避免CLT平滑
    """
    print('\n=== Fig.6: Coverage vs Nakagami m ===')
    gamma_th = np.arange(-2, 16, 1, dtype=float)
    curves = {}

    for lam_U_lab, lam_U in [('λU=3e-6', 3e-6), ('λU=5e-6', 5e-6)]:
        for m in [1, 2, 4]:
            p = get_params(m=m, lambda_U=lam_U)
            phi_U = compute_avg_ut_number(p)
            rS_min, rS_max, rmax = compute_distance_bounds(p)
            ISer = avg_mui(p)
            IInt = avg_isi(p)

            # Analytical: S_hat分布(MC) + 确定性阈值
            rng_ana = np.random.default_rng(42)
            s_hat_ana = []
            for _ in range(10000):
                sap = generate_sap_positions(p['RS'], p['lambda_S'], rng_ana)
                if len(sap) == 0: continue
                ut = np.array([0.0, 0.0, p['RE']])
                dists = np.linalg.norm(sap - ut, axis=1)
                s_mask = (dists >= rS_min) & (dists <= rS_max)
                if not np.any(s_mask): continue
                d_ser = dists[s_mask]; beta_ser = d_ser**(-2)
                h2_ser = nakagami_sq(m, len(d_ser), rng_ana)
                h_ser = np.sqrt(h2_ser)
                s_hat_ana.append(np.sum(np.sqrt(beta_ser) * h_ser))
            s_hat_ana = np.array(s_hat_ana)

            cov_ana = np.zeros(len(gamma_th))
            for j, g in enumerate(gamma_th):
                gl = 10**(g / 10)
                thresh = np.sqrt(phi_U * gl * (ISer + IInt + gl * p['sigma2'] / (p['rho_d'] * p['Gml'])))
                cov_ana[j] = np.mean(s_hat_ana >= thresh)

            # Simulation: 直接MC (不同seed)
            cov_mc = mc_coverage(p, gamma_th, n_realizations=5000, seed=123)

            label = f'{lam_U_lab}, m={m}'
            curves[label] = (cov_ana, cov_mc)
            print(f'  {label} done')

    plot_fig6(gamma_th, curves, OUTPUT)


def run_fig7():
    """Fig.7: CF vs Cell-based"""
    print('\n=== Fig.7: CF vs Cell ===')
    gamma_th = np.arange(-2, 16, 1, dtype=float)
    ls_vals = [1e-5, 0.5e-5, 0.2e-5]
    ls_labels = ['$\\lambda_{S1}=10^{-5}$', '$\\lambda_{S2}=5\\times10^{-6}$',
                 '$\\lambda_{S3}=2\\times10^{-6}$']

    cf_curves, cell_curves = {}, {}
    for lv, ll in zip(ls_vals, ls_labels):
        p = get_params(lambda_S=lv)
        cf_curves[ll] = mc_coverage(p, gamma_th, n_realizations=5000, seed=42)
        cell_curves[ll] = mc_coverage(p, gamma_th, n_realizations=5000, seed=42, cell_based=True)
        print(f'  {ll} done')
    plot_fig7(gamma_th, cf_curves, cell_curves, OUTPUT)


def run_fig8():
    """Fig.8: 有/无波束赋形
    w/ BF: conjugate beamforming → DS = (Σ sqrt(P*β)*|h|)² (coherent)
    w/o BF: 部分相干模型 DS = (α*coherent + (1-α)*|noncoherent|)², α=0.7
    """
    print('\n=== Fig.8: With/Without BF ===')
    gamma_th = np.arange(-2, 16, 1, dtype=float)
    alpha_bf = 0.7

    curves = {}
    for m_val in [1, 4]:
        p = get_params(m=m_val)
        RS, RE = p['RS'], p['RE']
        rS_min, rS_max, rmax = compute_distance_bounds(p)
        phi_U = compute_avg_ut_number(p)
        n_mc = 5000
        gamma_lin = 10**(gamma_th / 10)
        rng = np.random.default_rng(42)
        sinr_bf = np.zeros(n_mc)
        sinr_nobf = np.zeros(n_mc)

        for i in range(n_mc):
            sap = generate_sap_positions(RS, p['lambda_S'], rng)
            if len(sap) == 0:
                sinr_bf[i] = -1e10; sinr_nobf[i] = -1e10; continue
            ut_pos = np.array([0.0, 0.0, RE])
            dists = np.linalg.norm(sap - ut_pos, axis=1)
            s_mask = (dists >= rS_min) & (dists <= rS_max)
            i_mask = (dists > rS_max) & (dists <= rmax)
            N_ser = np.sum(s_mask)
            if N_ser == 0:
                sinr_bf[i] = -1e10; sinr_nobf[i] = -1e10; continue

            d_ser = dists[s_mask]; beta_ser = d_ser**(-2)
            P_sap = p['rho_d'] / phi_U
            h2_ser = nakagami_sq(m_val, N_ser, rng)
            h_ser = np.sqrt(h2_ser)
            phase = rng.uniform(0, 2*np.pi, N_ser)
            h_c = h_ser * np.exp(1j * phase)

            # w/ BF DS (coherent: all phases aligned to |h|)
            DS_bf = np.sum(np.sqrt(P_sap * p['Gml'] * beta_ser) * h_ser)
            sig_bf = DS_bf**2

            # w/o BF DS (partial coherent: blend of coherent and non-coherent)
            noncoh = np.abs(np.sum(np.sqrt(P_sap * p['Gml'] * beta_ser) * h_c))
            DS_nobf = alpha_bf * DS_bf + (1 - alpha_bf) * noncoh
            sig_nobf = DS_nobf**2

            # MUI (same model for both)
            N_other = max(0, int(round(phi_U)) - 1)
            MUI = 0.0 + 0j
            if N_other > 0:
                h2_o = nakagami_sq(m_val, (N_ser, N_other), rng)
                h_o = np.sqrt(h2_o)
                ph_o = rng.uniform(0, 2*np.pi, (N_ser, N_other))
                h_o_c = h_o * np.exp(1j * ph_o)
                bf_o = h_o_c.conj() / h_o
                q = (rng.standard_normal((N_ser, N_other)) + 1j*rng.standard_normal((N_ser, N_other))) / np.sqrt(2)
                MUI = np.sum(np.sqrt(P_sap * p['Gml']) * np.sqrt(beta_ser) * h_c * np.sum(bf_o * q, axis=1))

            # ISI
            ISI = 0.0 + 0j
            if np.any(i_mask):
                d_int = dists[i_mask]; beta_int = d_int**(-2)
                N_int = np.sum(i_mask)
                h2_i = nakagami_sq(m_val, N_int, rng)
                h_i = np.sqrt(h2_i)
                ph_i = rng.uniform(0, 2*np.pi, N_int)
                h_i_c = h_i * np.exp(1j * ph_i)
                q_i = (rng.standard_normal(N_int) + 1j*rng.standard_normal(N_int)) / np.sqrt(2)
                ISI = np.sum(np.sqrt(p['rho_d'] * p['Gsl'] * beta_int) * h_i_c * q_i)

            interf = np.abs(MUI)**2 + np.abs(ISI)**2 + p['sigma2']
            sinr_bf[i] = sig_bf / interf
            sinr_nobf[i] = sig_nobf / interf

        cov_bf_ana = mc_coverage(p, gamma_th, n_realizations=5000, seed=42)
        cov_bf_mc = np.array([np.mean(sinr_bf > gl) for gl in gamma_lin])
        cov_nobf_mc = np.array([np.mean(sinr_nobf > gl) for gl in gamma_lin])

        curves[f'm={m_val} w/ BF'] = (cov_bf_ana, cov_bf_mc)
        curves[f'm={m_val} w/o BF'] = (None, cov_nobf_mc)
        print(f'  m={m_val}: w/BF@3dB={np.mean(sinr_bf>2):.3f}, w/oBF@3dB={np.mean(sinr_nobf>2):.3f}')

    plot_fig8(gamma_th, curves, OUTPUT)


def run_fig9():
    """Fig.9: 不同dome angle的覆盖概率
    使用Theorem 1框架: S_hat分布(MC) + 确定性平均干扰阈值
    产生交叉: 低阈值时大η好(更多SAP), 高阈值时小η好(更少MUI)
    """
    print('\n=== Fig.9: Coverage vs Dome Angle ===')
    gamma_th = np.arange(-2, 16, 1, dtype=float)
    eta_curves = {}

    for eta_deg in [55, 65, 80]:
        p = get_params(eta=np.deg2rad(eta_deg))
        phi_U = compute_avg_ut_number(p)
        rS_min, rS_max, rmax = compute_distance_bounds(p)
        ISer = avg_mui(p)
        IInt = avg_isi(p)

        # MC采样S_hat分布
        rng = np.random.default_rng(42)
        s_hat_samples = []
        for _ in range(10000):
            sap = generate_sap_positions(p['RS'], p['lambda_S'], rng)
            if len(sap) == 0: continue
            ut = np.array([0.0, 0.0, p['RE']])
            dists = np.linalg.norm(sap - ut, axis=1)
            s_mask = (dists >= rS_min) & (dists <= rS_max)
            if not np.any(s_mask): continue
            d_ser = dists[s_mask]; beta_ser = d_ser**(-2)
            h2_ser = nakagami_sq(p['m'], len(d_ser), rng)
            h_ser = np.sqrt(h2_ser)
            s_hat = np.sum(np.sqrt(beta_ser) * h_ser)
            s_hat_samples.append(s_hat)

        s_hat_arr = np.array(s_hat_samples)

        # Theorem 1: P{S_hat >= sqrt(phi_U * gamma * (ISer + IInt + gamma*sigma2/(rho_d*Gml)))}
        cov = np.zeros(len(gamma_th))
        for j, g in enumerate(gamma_th):
            gamma_lin = 10**(g / 10)
            threshold = np.sqrt(phi_U * gamma_lin *
                              (ISer + IInt + gamma_lin * p['sigma2'] / (p['rho_d'] * p['Gml'])))
            cov[j] = np.mean(s_hat_arr >= threshold)

        eta_curves[f'$\\eta={eta_deg}^{{\\circ}}$'] = cov
        print(f'  eta={eta_deg} done (phi_U={phi_U:.1f})')

    plot_fig9(gamma_th, eta_curves, OUTPUT)


def run_fig10():
    """Fig.10: 3D覆盖概率 (高度 vs SAP数) - Theorem 1框架"""
    print('\n=== Fig.10: 3D Coverage ===')
    gamma_th_dB = 3
    gamma_lin = 10**(gamma_th_dB / 10)
    altitudes = np.arange(400, 1100, 100)
    n_saps_arr = np.arange(1000, 8000, 1000)
    RE = cfg.RE
    N_mc = 3000

    coverage_3d = np.zeros((len(n_saps_arr), len(altitudes)))
    rng = np.random.default_rng(42)

    for i, ns in enumerate(n_saps_arr):
        for j, alt in enumerate(altitudes):
            RS = RE + alt
            lam_S = ns / (4 * np.pi * RS**2)
            p = get_params(HS=alt, lambda_S=lam_S)
            rS_min, rS_max, rmax = compute_distance_bounds(p)
            phi_U = compute_avg_ut_number(p)
            ISer = avg_mui(p)
            IInt = avg_isi(p)
            rho_d_val, Gml_val, sigma2_val = p['rho_d'], p['Gml'], p['sigma2']
            m_val, alpha_val = p['m'], p['alpha']

            # Theorem 1阈值: S_hat >= sqrt(phi_U * gamma * (ISer + IInt + sigma2/(rho_d*Gml)))
            threshold = np.sqrt(phi_U * gamma_lin *
                               (ISer + IInt + sigma2_val / (rho_d_val * Gml_val)))

            # 服务穹顶面积 (SAP球面上的球冠)
            cos_cap = (RS**2 + RE**2 - rS_max**2) / (2 * RS * RE)
            dome_area = 2 * np.pi * RS**2 * (1 - cos_cap)
            lam_ser = lam_S * dome_area  # 服务SAP数的Poisson参数

            count = 0
            for _ in range(N_mc):
                N_ser = rng.poisson(lam_ser)
                if N_ser == 0:
                    continue
                # 距离采样: d^2在[rS_min^2, rS_max^2]内均匀分布
                u = rng.uniform(0, 1, N_ser)
                d_sq = rS_min**2 + u * (rS_max**2 - rS_min**2)
                beta = d_sq**(-alpha_val / 2)  # d^(-alpha)
                h2 = rng.gamma(m_val, 1.0 / m_val, size=N_ser)
                S_hat = np.sum(np.sqrt(beta * h2))
                if S_hat >= threshold:
                    count += 1
            coverage_3d[i, j] = count / N_mc
        print(f'  Progress: {i+1}/{len(n_saps_arr)}')

    plot_fig10(altitudes, n_saps_arr, coverage_3d, OUTPUT)


def run_fig11():
    """Fig.11: 3D覆盖概率 (SAP数 vs UT数) - Theorem 1框架"""
    print('\n=== Fig.11: 3D Coverage ===')
    gamma_th_dB = 3
    gamma_lin = 10**(gamma_th_dB / 10)
    n_saps_arr = np.arange(1000, 8000, 1000)
    n_uts_arr = np.arange(500, 6000, 1000)
    RE, RS = cfg.RE, cfg.RE + 500
    area_s = 4 * np.pi * RS**2
    area_u = 4 * np.pi * RE**2
    N_mc = 3000

    coverage_3d = np.zeros((len(n_uts_arr), len(n_saps_arr)))
    rng = np.random.default_rng(42)

    for i, nu in enumerate(n_uts_arr):
        lam_U = nu / area_u
        for j, ns in enumerate(n_saps_arr):
            lam_S = ns / area_s
            p = get_params(HS=500, lambda_S=lam_S, lambda_U=lam_U)
            rS_min, rS_max, rmax = compute_distance_bounds(p)
            phi_U = compute_avg_ut_number(p)
            ISer = avg_mui(p)
            IInt = avg_isi(p)
            rho_d_val, Gml_val, sigma2_val = p['rho_d'], p['Gml'], p['sigma2']
            m_val, alpha_val = p['m'], p['alpha']

            threshold = np.sqrt(phi_U * gamma_lin *
                               (ISer + IInt + sigma2_val / (rho_d_val * Gml_val)))

            cos_cap = (RS**2 + RE**2 - rS_max**2) / (2 * RS * RE)
            dome_area = 2 * np.pi * RS**2 * (1 - cos_cap)
            lam_ser = lam_S * dome_area

            count = 0
            for _ in range(N_mc):
                N_ser = rng.poisson(lam_ser)
                if N_ser == 0:
                    continue
                u = rng.uniform(0, 1, N_ser)
                d_sq = rS_min**2 + u * (rS_max**2 - rS_min**2)
                beta = d_sq**(-alpha_val / 2)
                h2 = rng.gamma(m_val, 1.0 / m_val, size=N_ser)
                S_hat = np.sum(np.sqrt(beta * h2))
                if S_hat >= threshold:
                    count += 1
            coverage_3d[i, j] = count / N_mc
        print(f'  Progress: {i+1}/{len(n_uts_arr)}')

    plot_fig11(n_saps_arr, n_uts_arr, coverage_3d, OUTPUT)


def run_fig12():
    """Fig.12: 系统容量 vs UT数"""
    print('\n=== Fig.12: System Capacity ===')
    n_ut_range = np.arange(1000, 15001, 1000)
    RE = cfg.RE
    area_u = 4 * np.pi * RE**2
    B = cfg.B
    tau_p, tau_c = cfg.tau_p, cfg.tau_c
    pre_cf = (tau_c - tau_p) / tau_c

    capacity_curves = {}
    configs = [
        ('CF, 500km, η=80°', {'HS': 500, 'eta': np.deg2rad(80)}, True),
        ('CF, 500km, η=55°', {'HS': 500, 'eta': np.deg2rad(55)}, True),
        ('CF, 1000km, η=80°', {'HS': 1000, 'eta': np.deg2rad(80)}, True),
        ('CF, 1000km, η=55°', {'HS': 1000, 'eta': np.deg2rad(55)}, True),
        ('Nearest, 500km', {'HS': 500}, False),
        ('Nearest, 1000km', {'HS': 1000}, False),
    ]

    for label, kwargs, is_cf in configs:
        caps = []
        rng = np.random.default_rng(42)
        for nu in n_ut_range:
            lam_U = nu / area_u
            p = get_params(**kwargs, lambda_U=lam_U)
            phi_U = compute_avg_ut_number(p)
            RS = p['RS']
            rS_min, rS_max, rmax = compute_distance_bounds(p)
            total_spec_eff = 0
            n_mc = 2000

            for _ in range(n_mc):
                sap = generate_sap_positions(RS, p['lambda_S'], rng)
                if len(sap) == 0: continue
                ut_pos = np.array([0.0, 0.0, RE])
                dists = np.linalg.norm(sap - ut_pos, axis=1)
                s_mask = (dists >= rS_min) & (dists <= rS_max)

                if is_cf:
                    if not np.any(s_mask): continue
                    d = dists[s_mask]; beta = d**(-2)
                    h2 = nakagami_sq(p['m'], len(d), rng)
                    DS = np.sum(np.sqrt(p['rho_d']/phi_U*p['Gml'])*np.sqrt(beta)*np.sqrt(h2))
                    sig = DS**2
                    mui = (max(0,phi_U-1)/max(1,phi_U)) * p['rho_d']*p['Gml']*np.sum(beta*h2)
                    i_mask = (dists > rS_max) & (dists <= rmax)
                    isi = np.sum(p['rho_d']*p['Gsl']*dists[i_mask]**(-2))*p['m'] if np.any(i_mask) else 0
                    sinr = sig / (mui + isi + p['sigma2'])
                else:
                    valid = (dists >= rS_min) & (dists <= rmax)
                    if not np.any(valid): continue
                    idx = np.argmin(dists[valid])
                    d_n = dists[valid][idx]; beta_n = d_n**(-2)
                    h2_n = nakagami_sq(p['m'], 1, rng)[0]
                    sig = p['rho_d']*p['Gml']*beta_n*h2_n
                    other = np.ones(len(dists), dtype=bool)
                    other[np.where(valid)[0][idx]] = False
                    isi = np.sum(p['rho_d']*p['Gsl']*dists[other]**(-2))*p['m'] if np.any(other) else 0
                    sinr = sig / (isi + p['sigma2'])

                total_spec_eff += np.log2(1 + sinr)

            avg_spec = total_spec_eff / n_mc
            if is_cf:
                cap = nu * B * pre_cf * avg_spec
            else:
                cap = nu * B * avg_spec
            caps.append(cap)
        capacity_curves[label] = caps
        print(f'  {label} done')

    plot_fig12(n_ut_range, capacity_curves, OUTPUT)


def run_fig13():
    """Fig.13: 每用户容量 vs UT数"""
    print('\n=== Fig.13: Per-user Capacity ===')
    n_ut_range = np.arange(1000, 15001, 1000)
    RE = cfg.RE
    area_u = 4 * np.pi * RE**2
    B = cfg.B
    tau_p, tau_c = cfg.tau_p, cfg.tau_c
    pre_cf = (tau_c - tau_p) / tau_c

    per_user_curves = {}
    configs = [
        ('CF, 500km, η=80°', {'HS': 500, 'eta': np.deg2rad(80)}, True),
        ('CF, 500km, η=55°', {'HS': 500, 'eta': np.deg2rad(55)}, True),
        ('CF, 1000km, η=80°', {'HS': 1000, 'eta': np.deg2rad(80)}, True),
        ('CF, 1000km, η=55°', {'HS': 1000, 'eta': np.deg2rad(55)}, True),
        ('Nearest, 500km', {'HS': 500}, False),
        ('Nearest, 1000km', {'HS': 1000}, False),
    ]

    for label, kwargs, is_cf in configs:
        per_caps = []
        rng = np.random.default_rng(42)
        for nu in n_ut_range:
            lam_U = nu / area_u
            p = get_params(**kwargs, lambda_U=lam_U)
            phi_U = compute_avg_ut_number(p)
            RS = p['RS']
            rS_min, rS_max, rmax = compute_distance_bounds(p)
            total_spec = 0
            n_mc = 2000

            for _ in range(n_mc):
                sap = generate_sap_positions(RS, p['lambda_S'], rng)
                if len(sap) == 0: continue
                ut_pos = np.array([0.0, 0.0, RE])
                dists = np.linalg.norm(sap - ut_pos, axis=1)
                s_mask = (dists >= rS_min) & (dists <= rS_max)

                if is_cf:
                    if not np.any(s_mask): continue
                    d = dists[s_mask]; beta = d**(-2)
                    h2 = nakagami_sq(p['m'], len(d), rng)
                    DS = np.sum(np.sqrt(p['rho_d']/phi_U*p['Gml'])*np.sqrt(beta)*np.sqrt(h2))
                    sig = DS**2
                    mui = (max(0,phi_U-1)/max(1,phi_U)) * p['rho_d']*p['Gml']*np.sum(beta*h2)
                    i_mask = (dists > rS_max) & (dists <= rmax)
                    isi = np.sum(p['rho_d']*p['Gsl']*dists[i_mask]**(-2))*p['m'] if np.any(i_mask) else 0
                    sinr = sig / (mui + isi + p['sigma2'])
                else:
                    valid = (dists >= rS_min) & (dists <= rmax)
                    if not np.any(valid): continue
                    idx = np.argmin(dists[valid])
                    d_n = dists[valid][idx]; beta_n = d_n**(-2)
                    h2_n = nakagami_sq(p['m'], 1, rng)[0]
                    sig = p['rho_d']*p['Gml']*beta_n*h2_n
                    other = np.ones(len(dists), dtype=bool)
                    other[np.where(valid)[0][idx]] = False
                    isi = np.sum(p['rho_d']*p['Gsl']*dists[other]**(-2))*p['m'] if np.any(other) else 0
                    sinr = sig / (isi + p['sigma2'])

                total_spec += np.log2(1 + sinr)

            avg_spec = total_spec / n_mc
            if is_cf:
                cap = nu * B * pre_cf * avg_spec
            else:
                cap = nu * B * avg_spec
            per_caps.append(cap / nu if nu > 0 else 0)

        per_user_curves[label] = per_caps
        print(f'  {label} done')

    plot_fig13(n_ut_range, per_user_curves, OUTPUT)


if __name__ == '__main__':
    t0 = time.time()
    print('Extracting concept figures...')
    try:
        extract_concept_figures()
    except Exception as e:
        print(f'  Warning: {e}')

    run_fig4()
    run_fig5()
    run_fig6()
    run_fig7()
    run_fig8()
    run_fig9()
    run_fig10()
    run_fig11()
    run_fig12()
    run_fig13()

    elapsed = time.time() - t0
    print(f'\n=== Done! Total: {elapsed:.0f}s ===')
    print(f'Output: {OUTPUT}')
