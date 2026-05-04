# WCL-2026 论文复现代码

## 论文信息

**标题**: Time-Division Spectrum Sharing and Coordination Between Beam-Hopping NGSO Satellites and Terrestrial Networks

**作者**: Kai Chang, Jianxiu Wang, Bingkun Liu, Linling Kuang (清华大学)

**发表**: IEEE Wireless Communications Letters, Vol. 15, 2026

## 项目结构

```
[020-WCL-2026].../
├── config.py              # 仿真参数配置 (Starlink星座、BH参数、时延等)
├── bhss_core.py           # BHSS 核心算法 (干扰关系、时间对齐)
├── simulation.py          # 主仿真模块 (Fig.5 服务容量, Fig.6 时间效率)
├── plotting.py            # IEEE 风格绘图
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
│   ├── fig5_service_capacity.png
│   ├── fig6_time_sync_efficiency.png
│   ├── fig1_architecture.png
│   ├── fig2_timeslot_mechanism.png
│   ├── fig3_schematic_map.png
│   └── fig4_beam_projections.png
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[020-WCL-2026]Time-Division_Spectrum_Sharing_and_Coordination_Between_Beam-Hopping_NGSO_Satellites_and_Terrestrial_Networks"
python run_reproduction.py
```

## 复现结果

### Fig. 5: 干扰小区平均服务容量
对比 4 种频谱共享方案 (BHSS, DSS, Fixed Freq Div, Interfered)。

| 用户数 | Interfered | Fixed Freq | DSS | BHSS |
|--------|-----------|------------|-----|------|
| 100 | 8.01 | 8.45 | 8.56 | 8.57 |
| 200 | 12.66 | 14.27 | 14.88 | 14.97 |
| 300 | 16.47 | 19.84 | 21.46 | 21.74 |
| 400 | 18.98 | 24.20 | 27.31 | 27.92 |
| 500 | 20.37 | 27.05 | 31.65 | 32.65 |

BHSS 相比 DSS 提升约 3% (论文 "up to 7%")。

### Fig. 6: 时间同步效率
对比 5 种时间对齐方法在不同时隙长度下的效率。

| T (ms) | Ideal | Proposed | TS-based | Gen Sync | Terr-prior |
|--------|-------|----------|----------|----------|------------|
| 1 | 100% | 98.0% | 89.0% | 92.9% | 99.0% |
| 5 | 100% | 96.5% | 81.0% | 95.0% | 98.5% |
| 10 | 100% | 96.2% | 71.0% | 95.0% | 98.2% |

Proposed 方法效率接近理想 (96-98%)。

## 核心算法

**Algorithm 1 (BHSS Time Alignment)**:
1. 基于 BH 模式和星历计算干扰关系
2. 扩展干扰时间段 (考虑时延变化和 GP)
3. 标记时隙状态 (可用/不可用/特殊时隙)
4. 特殊时隙配置: 边界时隙通过 ST_c2s/ST_s2c 符号部分复用

**简化说明**: 由于论文使用实际 LTE/NR 网络数据和复杂的多星 BH 调度算法 [10]，本复现采用半解析模型，参数根据论文仿真设置和结果描述校准。

## 依赖库

```bash
pip install numpy matplotlib
```
