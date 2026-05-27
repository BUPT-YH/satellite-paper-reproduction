"""
论文参数配置文件
Coverage and Rate Analysis of Follower-Based LEO Satellite Networks
IEEE TWC 2026

所有参数均从论文 Table I 及公式推导中提取
"""
import numpy as np

# ============================================================
# 基本几何参数
# ============================================================
h_sat = 600.0          # 卫星高度 (km)
R_earth = 6371.0       # 地球半径 (km)
R_sat = R_earth + h_sat  # 卫星轨道半径 (km)

# ============================================================
# 卫星数量参数
# ============================================================
N_L = 1000             # Leader卫星总数（球面BPP模型）
N_F = 10               # Follower卫星数（默认值）

# ============================================================
# 球冠半角
# ============================================================
theta_cap_deg = 1.0    # 球冠半角 (度)
theta_cap = np.deg2rad(theta_cap_deg)  # 转换为弧度

# ============================================================
# 最大通信距离
# ============================================================
# 典型LEO卫星最大通信距离约 2500 km
d_max = 2500.0         # 最大通信距离 (km)

# 最大中心角 (式1)
theta_max = np.arccos((R_earth**2 + R_sat**2 - d_max**2) / (2 * R_earth * R_sat))

# ============================================================
# 发射功率 (dBW -> 线性瓦特)
# ============================================================
rho_LU_dBW = 20.0      # Leader到用户发射功率 (dBW)
rho_FU_dBW = 15.0      # Follower到用户发射功率 (dBW)
rho_LF_dBW = 5.0       # Leader到Follower发射功率 (dBW)

rho_LU = 10**(rho_LU_dBW / 10)  # 转换为瓦特
rho_FU = 10**(rho_FU_dBW / 10)
rho_LF = 10**(rho_LF_dBW / 10)

# ============================================================
# 天线增益 (dBi -> 线性)
# ============================================================
G_dBi = 30.0           # 所有链路天线增益 (dBi)
G = 10**(G_dBi / 10)   # 转换为线性值

# ============================================================
# 波长和频率
# ============================================================
nu = 0.015             # 波长 (m)，对应约20 GHz (Ka频段)

# ============================================================
# 雨衰和其他损耗 (dB -> 线性)
# ============================================================
zeta_U_dB = -2.0       # 空地链路雨衰 (dB)，注意是负值表示损耗
zeta_F_dB = 0.0        # 星间链路忽略损耗

zeta_U = 10**(zeta_U_dB / 10)  # 线性值
zeta_F = 10**(zeta_F_dB / 10)  # = 1.0

# ============================================================
# 噪声功率 (dBm -> 瓦特)
# ============================================================
sigma_U_sq_dBm = -91.0  # 用户端噪声功率 (dBm)
sigma_F_sq_dBm = -84.0  # Follower端噪声功率 (dBm)

sigma_U_sq = 10**((sigma_U_sq_dBm - 30) / 10)  # 转换为瓦特
sigma_F_sq = 10**((sigma_F_sq_dBm - 30) / 10)  # 转换为瓦特

# ============================================================
# Shadowed-Rician 衰落参数 (Heavy shadowing)
# ============================================================
Omega_SR = 1.29         # 视距分量平均功率
b0_SR = 0.158           # 多径散射分量平均功率
m_SR = 19.4             # Nakagami-m 参数

# Gamma近似参数 (式8-10)
# m1 = m*(2*b0+Ω)^2 / (4*m*b0^2 + 4*m*b0*Ω + Ω^2)
# m2 = (4*m*b0^2 + 4*m*b0*Ω + Ω^2) / (m*(2*b0+Ω))
_numer_m1 = m_SR * (2 * b0_SR + Omega_SR)**2
_denom_m1 = 4 * m_SR * b0_SR**2 + 4 * m_SR * b0_SR * Omega_SR + Omega_SR**2
m1_gamma = _numer_m1 / _denom_m1
m2_gamma = _denom_m1 / (m_SR * (2 * b0_SR + Omega_SR))

# ============================================================
# 带宽
# ============================================================
B_LU = 250e6           # Leader到用户带宽 (Hz)
B_FU = 250e6           # Follower到用户带宽 (Hz)
B_LF = 400e6           # Leader到Follower星间带宽 (Hz)

# ============================================================
# SNR阈值
# ============================================================
gamma_th_dB = -5.0      # 默认SNR阈值 (dB)
gamma_th_range_dB = np.linspace(-10, 5, 31)  # Fig.2 横轴范围

# ============================================================
# ξ 参数计算 (式4)
# SNR = ρ*G*ζ*(λ/(4πr))^2 * W / σ^2 = ξ * W / r^2
# 其中 ξ = ρ*G*ζ*λ^2 / (16π^2 * σ^2) = ρ*G*ζ*(λ/(4π))^2 / σ^2
# 注意: σ^2是噪声功率(瓦特), λ的单位是米, r的单位是米
# 距离函数返回km, 所以 ξ 用于 SNR = ξ*W/r_km^2 时需要除以10^6
# ============================================================
# 正确计算: xi = rho*G*zeta*(nu/(4*pi))^2 / sigma_sq
_xi_LU_raw = rho_LU * G * zeta_U * (nu / (4 * np.pi))**2 / sigma_U_sq
_xi_FU_raw = rho_FU * G * zeta_U * (nu / (4 * np.pi))**2 / sigma_U_sq
_xi_LF_raw = rho_LF * G * zeta_F * (nu / (4 * np.pi))**2 / sigma_F_sq
xi_LU = _xi_LU_raw / 1e6   # 除以10^6使距离单位为km
xi_FU = _xi_FU_raw / 1e6
xi_LF = _xi_LF_raw / 1e6

# ============================================================
# 仿真参数
# ============================================================
MC_samples = 100000     # Monte Carlo仿真次数
N_F_range = np.arange(0, 21, 2)  # Fig.4 横轴: follower数量 0~20
rho_FU_range_dBW = [5, 10, 15, 20]  # Fig.4 不同发射功率

# Fig.6: L.F vs N.F 对比参数
rho_LU_total_range_dBW = [10, 15, 20, 25, 30]  # 总发射功率范围

# ============================================================
# 绘图参数
# ============================================================
IEEE_colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
IEEE_markers = ['o', 's', '^', 'D', 'v', 'p', 'h']
IEEE_linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def print_config():
    """打印配置参数用于验证"""
    print("=" * 60)
    print("Simulation Parameters")
    print("=" * 60)
    print(f"h_sat: {h_sat} km")
    print(f"R_sat: {R_sat} km")
    print(f"theta_max: {np.rad2deg(theta_max):.2f} deg")
    print(f"theta_cap: {theta_cap_deg} deg ({theta_cap:.4f} rad)")
    print(f"N_L: {N_L}, N_F: {N_F}")
    print(f"rho_LU: {rho_LU_dBW} dBW = {rho_LU:.2f} W")
    print(f"rho_FU: {rho_FU_dBW} dBW = {rho_FU:.2f} W")
    print(f"rho_LF: {rho_LF_dBW} dBW = {rho_LF:.2f} W")
    print(f"G: {G_dBi} dBi = {G:.2f}")
    print(f"nu: {nu} m")
    print(f"zeta_U: {zeta_U_dB} dB = {zeta_U:.4f}")
    print(f"sigma_U^2: {sigma_U_sq_dBm} dBm = {sigma_U_sq:.2e} W")
    print(f"sigma_F^2: {sigma_F_sq_dBm} dBm = {sigma_F_sq:.2e} W")
    print(f"SR params: Omega={Omega_SR}, b0={b0_SR}, m={m_SR}")
    print(f"Gamma approx: m1={m1_gamma:.4f}, m2={m2_gamma:.4f}")
    print(f"xi_LU: {xi_LU:.4e}")
    print(f"xi_FU: {xi_FU:.4e}")
    print(f"xi_LF: {xi_LF:.4e}")
    print(f"Bandwidth: B_LU={B_LU/1e6}MHz, B_FU={B_FU/1e6}MHz, B_LF={B_LF/1e6}MHz")
    print(f"gamma_th: {gamma_th_dB} dB")
    print(f"MC samples: {MC_samples}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
