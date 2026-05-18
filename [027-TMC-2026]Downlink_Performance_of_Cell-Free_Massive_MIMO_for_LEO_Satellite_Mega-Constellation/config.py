"""
仿真参数配置
论文: Downlink Performance of Cell-Free Massive MIMO for LEO Satellite Mega-Constellation
期刊: IEEE TMC, 2026
"""

import numpy as np

# ===== 物理常数 =====
c_km = 3e5          # 光速 (km/s)
c_m = 3e8           # 光速 (m/s)

# ===== Table I 默认参数 =====
RE = 6371.393       # 地球半径 (km)
HS = 500            # 轨道高度 (km)
RS = RE + HS        # SAP球面半径 (km)

# 密度参数
lambda_S = 1e-5     # SAP密度 (/km²)
lambda_U = 3e-6     # UT密度 (/km²)

# 频率
fc = 2e9            # 载波频率 (Hz) = 2 GHz
B = 30e6            # 带宽 (Hz) = 30 MHz

# 功率 (转换为瓦特)
rho_d_dBm = 33      # 下行发射功率 (dBm)
rho_p_dBm = 30      # 导频发射功率 (dBm)
sigma2_dBm = -100   # 噪声功率 (dBm)

rho_d = 10**(rho_d_dBm / 10) * 1e-3   # W
rho_p = 10**(rho_p_dBm / 10) * 1e-3   # W
sigma2 = 10**(sigma2_dBm / 10) * 1e-3  # W

# 天线增益 (dBi)
Gml_t_dBi = 30      # SAP主瓣发射增益
Gsl_t_dBi = 20      # SAP旁瓣发射增益
Gr_dBi = 0          # UT接收增益

# 有效天线增益 (Eq.2): G = Gt * Gr * (c/(4πfc))²
# 使用 c 单位 km/s, d 单位 km, 所以 (c/(4πfc))² 单位 km²
freq_factor = (c_km / (4 * np.pi * fc))**2  # km²

Gml = 10**(Gml_t_dBi / 10) * 10**(Gr_dBi / 10) * freq_factor  # km²
Gsl = 10**(Gsl_t_dBi / 10) * 10**(Gr_dBi / 10) * freq_factor  # km²

# 信道参数
m_nakagami = 2       # Nakagami衰落参数 (默认)
alpha = 2            # 路径损耗指数
beta0 = 1.0          # 参考距离路径损耗 (已在G中包含频率因子)

# 空分复用参数
eta = np.deg2rad(75) # 圆顶角 (默认75°)
tau_p = 200          # 导频长度
tau_c = 500          # 相干块长度

# ===== 辅助函数 =====
def get_params(**kwargs):
    """获取参数字典，可覆盖默认值"""
    p = {
        'RE': RE, 'RS': RS, 'HS': HS,
        'lambda_S': lambda_S, 'lambda_U': lambda_U,
        'fc': fc, 'B': B,
        'rho_d': rho_d, 'rho_p': rho_p, 'sigma2': sigma2,
        'Gml': Gml, 'Gsl': Gsl,
        'm': m_nakagami, 'alpha': alpha, 'beta0': beta0,
        'eta': eta,
        'tau_p': tau_p, 'tau_c': tau_c,
    }
    p.update(kwargs)
    # 自动更新 RS
    if 'HS' in kwargs and 'RS' not in kwargs:
        p['RS'] = p['RE'] + p['HS']
    return p


def compute_distance_bounds(p):
    """计算距离边界"""
    RS, RE, eta = p['RS'], p['RE'], p['eta']
    rS_min = RS - RE
    sin_eta = np.sin(eta)
    cos_eta = np.cos(eta)
    rS_max = np.sqrt(RS**2 - RE**2 * sin_eta**2) - RE * cos_eta
    rmax = np.sqrt(RS**2 - RE**2)
    return rS_min, rS_max, rmax


def compute_Hv(p):
    """计算最短垂直距离 (Eq.11)"""
    RS, RE, eta = p['RS'], p['RE'], p['eta']
    if eta >= np.pi / 2:
        return 0.0
    cos_eta = np.cos(eta)
    Hv = cos_eta**2 * (np.sqrt(RE**2 + (RS**2 - RE**2) / cos_eta**2) - RE)
    return Hv


def compute_avg_ut_number(p):
    """计算每个SAP服务区域内的平均UT数 (Eq.12)"""
    RS, RE, eta = p['RS'], p['RE'], p['eta']
    lambda_U = p['lambda_U']
    sin_eta = np.sin(eta)
    cos_eta = np.cos(eta)
    term = RE - RE**2 * sin_eta**2 / RS - RE * np.sqrt(RS**2 - RE**2 * sin_eta**2) * cos_eta / RS
    avg_ut = 2 * np.pi * RE * lambda_U * term
    return avg_ut


def compute_G(Gml_t_dBi_val=None, Gsl_t_dBi_val=None, Gr_dBi_val=None, fc_val=None):
    """计算有效天线增益"""
    if Gml_t_dBi_val is None:
        Gml_t_dBi_val = Gml_t_dBi
    if Gsl_t_dBi_val is None:
        Gsl_t_dBi_val = Gsl_t_dBi
    if Gr_dBi_val is None:
        Gr_dBi_val = Gr_dBi
    if fc_val is None:
        fc_val = fc
    ff = (c_km / (4 * np.pi * fc_val))**2
    gml = 10**(Gml_t_dBi_val / 10) * 10**(Gr_dBi_val / 10) * ff
    gsl = 10**(Gsl_t_dBi_val / 10) * 10**(Gr_dBi_val / 10) * ff
    return gml, gsl
