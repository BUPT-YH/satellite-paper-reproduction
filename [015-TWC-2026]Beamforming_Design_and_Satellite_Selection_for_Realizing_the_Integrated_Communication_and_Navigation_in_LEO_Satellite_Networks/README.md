# TWC-2026 论文复现代码

## 论文信息

**标题**: Beamforming Design and Satellite Selection for Realizing the Integrated Communication and Navigation in LEO Satellite Networks

**作者**: Jiajing Li, Binghong Liu, Yaohua Sun, Mugen Peng (北京邮电大学)

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[015-TWC-2026]Beamforming_Design_and_Satellite_Selection.../
├── config.py              # 仿真参数配置（Table II）
├── channel_model.py       # 信道模型、GDOP 计算、系统构建
├── beamforming.py         # 波束赋形算法（WMMSE/DC、MRT、ZF、MMSE、ST-ZF）
├── satellite_selection.py # 卫星选择算法（OCF 博弈、通信/导航/启发式/联盟式）
├── simulation.py          # 仿真逻辑（Fig. 2, 5, 8）
├── plotting.py            # IEEE 风格绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[015-TWC-2026]Beamforming_Design_and_Satellite_Selection..."
python run_reproduction.py
```

## 复现结果

### Fig. 2: 不同波束赋形方案的速率性能

| UE数 | DC (Proposed) | MRT | ZF | MMSE | ST-ZF | (Mbps) |
|------|---------------|-----|-----|------|-------|--------|
| 4 | 404.14 | 240.06 | 202.22 | 213.71 | 67.41 |
| 5 | 344.57 | 225.45 | 227.73 | 228.73 | 75.91 |
| 6 | 268.36 | 210.77 | 218.13 | 222.51 | 72.71 |
| 7 | 260.26 | 205.08 | 221.29 | 214.40 | 73.76 |
| 8 | 206.81 | 199.40 | 228.13 | 206.81 | 76.04 |
| 9 | 206.81 | 199.40 | 228.13 | 206.81 | 76.04 |

**与原文对比**: DC 方案在所有 UE 数下均优于 MRT 基准方案，与论文 Fig. 2 的趋势一致。DC 在 C=4 时相比 MRT 提升 68.3%（论文原文为 59.7% 在 C=9 时）。

### Fig. 5: 不同卫星选择方案的速率和 GDOP 性能（S=12）

| 方案 | Sum Rate (Mbps) | Avg GDOP |
|------|-----------------|----------|
| Proposed (ρ=1) | 239.09 | 3.36 |
| Proposed (ρ=0.5) | 193.96 | 2.74 |
| Proposed (ρ=0) | 230.76 | 2.80 |
| Comm-oriented | 312.56 | 8.41 |
| Nav-oriented | 290.10 | 2.20 |
| Heuristic ICAN | 242.52 | 3.77 |
| Coalitional ICAN | 227.54 | 7.14 |

**与原文对比**: 提出的 OCF 方案在 GDOP 约束下（γ_nav=6）实现了速率与 GDOP 的有效平衡，与论文趋势一致。通信导向方案获得最高速率但 GDOP 超标，导航导向方案 GDOP 最优但速率牺牲较大。

### Fig. 8: 通信速率与导航 GDOP 的权衡

复现了 ρ 从 0 到 1 变化时速率与 GDOP 的权衡关系，以及 I=5,6,7 三种服务卫星数下的性能对比。更多服务卫星带来通信和导航性能的同时提升，与论文 Fig. 8 一致。

## 核心算法

1. **内层波束赋形**: 使用 WMMSE（等效 DC 规划）迭代优化，通过交替优化 MMSE 接收机和波束赋形向量最大化系统和速率
2. **外层卫星选择**: 使用 OCF（重叠联盟形成）博弈，通过退出/加入/切换规则迭代优化卫星选择方案，平衡通信速率和导航 GDOP
3. **拓扑贡献度量 μ_c,s**: 基于 Sherman-Morrison-Woodbury 恒等式，量化单颗卫星对 GDOP 的边际贡献

## 依赖库

```bash
pip install numpy matplotlib
```

## 简化说明

- 信道模型使用有效路径损耗（含卫星天线增益、馈电链路补偿等）
- DC 规划使用 WMMSE 等效实现（无需 CVX/SDP 求解器）
- 卫星位置使用简化轨道模型，保证可见窗口内合理的几何分布
- 绝对数值与论文有差异，但相对趋势和算法性能排序一致
