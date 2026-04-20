# TWC-2026 论文复现代码

## 论文信息

**标题**: Space-Time Beamforming for LEO Satellite Communications: Enabling Extremely Narrow Beams

**作者**: Jungbin Yim, Jinseok Choi, Jeonghun Park, Ian P. Roberts, Namyoon Lee

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[012-TWC-2026]Space-Time_Beamforming_for_LEO_Satellite_Communications_Enabling_Extremely_Narrow_Beams/
├── config.py              # 仿真参数配置
├── channel.py             # 信道模型（UPA、Shadowed-Rician、空时信道）
├── beamforming.py         # 波束赋形算法（MRT/ZF/SLNR/ST-ZF/ST-SLNR/TDMA）
├── simulation.py          # 主仿真逻辑
├── plotting.py            # 绘图模块（IEEE 期刊风格）
├── run_reproduction.py    # 快速复现脚本
├── output/                # 输出图表
├── README.md
└── [012-TWC-2026]...pdf   # 原文
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[012-TWC-2026]Space-Time_Beamforming_for_LEO_Satellite_Communications_Enabling_Extremely_Narrow_Beams"
python run_reproduction.py
```

## 复现目标

### Fig. 5: 部分连接网络 SE vs P (K=3)
对比 MRT、ZF、SLNR (MMSE)、TDMA、ST-ZF 在不同发射功率下的和频谱效率。

核心发现：ST-ZF 在最优重传间隔 τ* 下获得比 TDMA 高 3 dB 的 SNR 增益。

### Fig. 7: 全连接网络 SE vs P (M=3, K=4)
对比 MRT、SLNR、TDMA、ST-SLNR 在全连接干扰网络中的性能。

核心发现：ST-SLNR 利用多普勒域时间特征，在干扰密集场景显著优于 TDMA。

### Fig. 8: ST-SLNR SE vs K 和 M (P=40 dBm)
展示重复次数 M 与用户数 K 的 trade-off。

核心发现：M 存在最优值（如 K=4 时 M*=3），过大会因 pre-log 损失抵消干扰抑制增益。

## 核心算法

### ST-ZF (Space-Time Zero-Forcing)
1. 选择最优重传间隔 τ* = 1/(2Δf)，使期望信道与干扰信道在时间域正交
2. 使用 MRT 波束赋形（正交性保证零泄漏）
3. 获得 4N 阵列增益（M=2, N=64），相比 TDMA 的 N 增益提升 3 dB

### ST-SLNR (Space-Time SLNR)
1. 联合优化 precoding vector f、重传间隔 τ、重复次数 M
2. τ 通过网格搜索最大化 SLNR
3. M 通过单调递增搜索最大化和频谱效率

## 仿真参数

| 参数 | 值 |
|------|-----|
| UPA 天线数 | Nx=Ny=8, N=64 |
| 载频 | 1.9925 GHz |
| 带宽 | 5 MHz |
| 轨道高度 | 530 km |
| 路径损耗指数 | 2 |
| 噪声功率谱密度 | -174 dBm/Hz |
| 多普勒范围 | [-50, 50] kHz |
| 多径数 | L=3, δ=0.5 |
| 衰落模型 | Shadowed-Rician（平均阴影） |

## 复现说明

- 用户共址模型：从同一卫星看不同用户的 AoA 几乎相同（差异 < 0.5°），模拟实际 LEO 波束覆盖场景
- 空时信道通过 Kronecker 积构造：h = b(f, τ) ⊗ a(θ, φ)
- τ 使用精确最优值（不量化到采样周期），保证时间域正交性
- Monte Carlo 仿真 200 次信道实现取平均

## 依赖库

```bash
pip install numpy matplotlib scipy
```
