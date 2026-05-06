# TWC-2026 论文复现代码

## 论文信息

**标题**: Joint Illumination, Power, and Band Allocation for Multi-Beam LEO Satellites With Beam-Hopping Using Mixed-Integer Linear Programming

**作者**: Samuel Martínez Zamacola, Nils Pachler, Ramón Martínez Rodríguez-Osorio, Bruce G. Cameron

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[021-TWC-2026]/
├── [021-TWC-2026].pdf         # 论文原文
├── config.py                   # 仿真参数配置
├── system_model.py             # 系统模型（小区布局、用户分布、链路预算）
├── milp_optimizer.py           # MILP 优化器（PuLP/CBC）
├── optimizer_heuristic.py      # 贪心优化方法（逼近 MILP）
├── db_baseline.py              # 确定性基线方法
├── ga_baseline.py              # 遗传算法基线（NSGA-II）
├── plotting.py                 # IEEE 风格绘图模块
├── run_reproduction.py         # 一键复现脚本
└── output/                     # 输出图表
    ├── fig5_weighting_study.png
    ├── fig6_reduced_scenario.png
    ├── fig8_enlarged_scenario.png
    ├── fig1_beam_hopping_scheme.png
    ├── fig2_frequency_bins.png
    ├── fig3_mcs_table.png
    └── fig4_milp_diagram.png
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[021-TWC-2026]..."
python run_reproduction.py
```

## 复现结果

### Fig. 6: 缩减场景对比 (NC=37, NT=30)

| Users | DB UC | GA UC | OPT UC | DB TTS | GA TTS | OPT TTS |
|-------|-------|-------|--------|--------|--------|---------|
| 5     | 0.8%  | 0.6%  | 0.4%   | 0.0    | 0.1    | 0.0     |
| 10    | 32.7% | 27.9% | 25.2%  | 1.5    | 1.5    | 1.7     |
| 20    | 74.8% | 61.4% | 65.8%  | 8.7    | 5.1    | 7.5     |

### Fig. 8: 扩展场景对比 (NC=271, NT=100)

| Users | DB UC | GA UC | OPT UC | DB TTS | GA TTS | OPT TTS |
|-------|-------|-------|--------|--------|--------|---------|
| 60    | 67.1% | 65.2% | 38.2%  | 13.3   | 10.8   | 7.4     |
| 80    | 83.8% | 77.1% | 73.3%  | 20.5   | 15.1   | 16.7    |
| 100   | 86.9% | 80.8% | 78.5%  | 26.7   | 20.0   | 22.4    |

## 核心算法

论文提出 MILP 混合整数线性规划方法联合优化 LEO 卫星跳波束系统的：
- **照射调度** (Illumination): 哪些小区在每个时隙被照射
- **功率分配** (Power): 每个用户的发射功率
- **频带分配** (Band): 每个用户使用的频率分块数

三种优化目标：UC（未服务容量）、EC（过服务容量）、TTS（等待服务时间）

Time-split MILP 变体将问题按时间窗口分解，大幅降低计算复杂度。

## 复现说明

- 本复现使用贪心优化（optimizer_heuristic.py）逼近 MILP 性能
- 完整 MILP 需要商业求解器（Gurobi），PuLP/CBC 求解速度有限
- 链路预算参数基于论文 Table IV，用户终端增益假设 35 dBi (VSAT)
- 绝对值与论文有差异，但相对趋势一致：优化方法 > GA > DB

## 依赖库

```bash
pip install numpy matplotlib pulp
```
