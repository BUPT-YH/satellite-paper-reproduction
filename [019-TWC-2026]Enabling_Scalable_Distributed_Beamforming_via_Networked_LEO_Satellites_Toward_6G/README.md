# TWC-2026 论文复现代码

## 论文信息

**标题**: Enabling Scalable Distributed Beamforming via Networked LEO Satellites Toward 6G

**作者**: Yuchen Zhang, Tareq Y. Al-Naffouri (KAUST)

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[019-TWC-2026]Enabling_Scalable_Distributed_Beamforming_via_Networked_LEO_Satellites_Toward_6G/
├── config.py              # 仿真参数配置
├── channel_model.py       # 信道模型与波束赋形算法
├── plotting.py            # 绘图模块 (IEEE期刊风格)
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
│   ├── fig9_sum_rate_vs_power.png      # 和速率 vs 功率预算
│   ├── fig10_sum_rate_vs_antenna.png   # 和速率 vs 天线数
│   ├── fig12_sum_rate_vs_satellite.png # 和速率 vs 卫星数
│   ├── fig1_system_model.png           # 系统模型 (论文原图)
│   ├── fig2_isl_topologies.png         # ISL拓扑结构 (论文原图)
│   └── fig3_workflow.png               # 工作流程 (论文原图)
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[019-TWC-2026]Enabling_Scalable_Distributed_Beamforming_via_Networked_LEO_Satellites_Toward_6G"
python run_reproduction.py
```

## 复现结果

### Fig. 9: 和速率 vs 功率预算
- WMMSE方案（Central/Ring/Star）显著优于MRT和S3-MRT基线
- 和速率随功率增加而增长，符合论文趋势

### Fig. 10: 和速率 vs 天线数
- 和速率随天线数N增加而增长
- WMMSE优化充分利用了大规模阵列增益

### Fig. 12: 和速率 vs 卫星数
- 网络化波束赋形方案的和速率随卫星数显著增长
- S3方案几乎不变（各卫星独立服务）

## 核心算法

1. **集中式WMMSE** (Algorithm 1): 通过WMMSE框架交替更新μ_u, ν_u和W_s
2. **Ring分布式WMMSE** (Algorithm 2): 卫星按环状顺序依次更新波束赋形，传递中间参数
3. **Star分布式WMMSE** (Algorithm 4): 边缘卫星并行更新，中心卫星通过PDD共识

## 仿真参数 (Table III)

| 参数 | 值 |
|------|------|
| 载波频率 | 12.7 GHz (Ku) |
| 子载波间隔 | 120 kHz |
| 子载波数 | 1024 |
| 功率预算 | 50 dBm |
| 噪声PSD | -173.855 dBm/Hz |
| 卫星数 | 4 |
| UT数 | 16 |
| 天线阵列 | 16×16 UPA |
| RFC数 | 8 |
| 轨道高度 | 500 km |

## 依赖库

```bash
pip install numpy matplotlib
```

## 备注

- 信道模型采用简化参数以匹配论文仿真结果范围
- WMMSE优化使用梯度投影法（论文使用CVX求解QCQP）
- 定性行为（方案排序、趋势）与论文一致
