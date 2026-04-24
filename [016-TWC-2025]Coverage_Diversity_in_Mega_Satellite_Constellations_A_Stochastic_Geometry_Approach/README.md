# TWC-2025 论文复现代码

## 论文信息

**标题**: Coverage Diversity in Mega Satellite Constellations: A Stochastic Geometry Approach

**作者**: Bassel Al Homssi, Ahmed Al-Amri, Jie Ding, Chiu Chun Chan, Jawad Al Attari, Mustafa A. Kishk, Jinho Choi, Akram Al-Hourani

**发表**: IEEE Transactions on Wireless Communications, Vol. 24, No. 11, November 2025

## 项目结构

```
[016-TWC-2025]Coverage_Diversity_in_Mega_Satellite_Constellations_A_Stochastic_Geometry_Approach/
├── config.py              # 仿真参数配置 (Table II)
├── stochastic_geometry.py # 核心算法: 随机几何覆盖概率模型
├── plotting.py            # IEEE 期刊风格绘图模块
├── simulation.py          # 主仿真逻辑
├── run_fast.py            # 一键快速复现脚本
├── run_reproduction.py    # 完整复现脚本 (含 MC)
├── output/                # 输出图表
│   ├── fig1_system_model.png         # 场景图: 双壳层卫星星座
│   ├── fig3_interferer_geometry.png  # 场景图: 干扰几何模型
│   ├── fig4_coverage_comparison.png  # 覆盖概率对比
│   ├── fig5_starlink_comparison.png  # Starlink Phase 2 对比
│   └── fig6_user_intensity.png       # 用户密度影响
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[016-TWC-2025]Coverage_Diversity_in_Mega_Satellite_Constellations_A_Stochastic_Geometry_Approach"
python run_fast.py
```

## 复现结果

### Fig.4: 覆盖概率对比 (理论 vs MC)
- **卫星选择分集**: 理论模型与 Monte Carlo 仿真精确匹配
- **合并分集 (MRC)**: 多壳层架构显著提升覆盖概率
- 单壳层 (N=900, R=600km) vs 三壳层 (N={900,400,100}, R={600,900,1200}km)

### Fig.5: Starlink Phase 2 对比
- 利用论文理论模型预测 Starlink Phase 2 三壳层网络性能
- 配置: Nm={2493,2478,2547}, Rm={335.9,340.8,345.6}km

### Fig.6: 用户密度影响 (γo = -20 dB)
- 分析不同壳层数对覆盖概率的影响
- 合并分集在多壳层架构下提供显著增益

## 核心算法

论文的核心贡献是基于随机几何的覆盖概率解析模型：

1. **卫星选择分集**: P_SS = 1 - exp(-N̄·(1-p_out)), 利用 Poisson 点过程展开
2. **MRC 合并分集**: 基于信号功率和的 CDF，通过 Laplace 变换逆变换求解
3. **多壳层扩展**: 各壳层独立计算后通过乘积形式组合

## 关键公式

- 路径损耗: l(φ) = (c/(4π·fc·d(φ)))²
- LoS 概率: p_LoS(φ) = exp(-β·sin(φ)/(cos(φ)-α_m))
- 平均干扰: Ī_m = 2π·λ·R²⊕·ρ_t·G_t·G_r·∫l·ζ̄·sin(φ)dφ
- 阴影衰落: ζ[dB] ~ p_LoS·N(-μ_LoS,σ²_LoS) + p_nLoS·N(-μ_nLoS,σ²_nLoS)

## 仿真参数 (Table II)

| 参数 | 值 | 单位 |
|------|-----|------|
| R_⊕ | 6731 | km |
| fc | 2 | GHz |
| β | 0.4 | - |
| μ_LoS, μ_nLoS, σ_LoS, σ_nLoS | 0.4, 0, 1, 5.2 | dB |
| ρ_t | 10 | dBm |
| G_t, G_r | 3, 2 | dBi |
| Ws | -160 | dBm |
| λo | 1 user/100 km² | - |
| Do | 25% | - |

## 依赖库

```bash
pip install numpy matplotlib scipy Pillow PyMuPDF
```

## 复现笔记

- 选择分集的理论模型与 MC 仿真高度吻合，验证了论文 Eq.(22) 的正确性
- 合并分集的 Euler 逆 Laplace 变换数值稳定性需要适当选择精度参数 L
- 论文使用 BPP (二项式点过程) 近似为 PPP (Poisson 点过程) 以获得解析解
- 大尺度干扰近似 (Theorem 1) 使得分析大幅简化，且在巨型星座中精度很高
