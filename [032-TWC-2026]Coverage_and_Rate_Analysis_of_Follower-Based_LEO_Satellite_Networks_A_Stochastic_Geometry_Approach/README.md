# TWC-2026 论文复现代码

## 论文信息

**标题**: Coverage and Rate Analysis of Follower-Based LEO Satellite Networks: A Stochastic Geometry Approach

**作者**: Juanjuan Ru, Ruibo Wang, Mohamed-Slim Alouini (KAUST)

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

**DOI**: 10.1109/TWC.2026.3666549

## 项目结构

```
[032-TWC-2026]Coverage_and_Rate_Analysis.../
├── config.py                    # 仿真参数配置（Table I 全部参数）
├── stochastic_geometry.py       # 核心随机几何算法模块
│   ├── 接触角PDF: Leader-user (Lemma 1), Follower-user (Lemma 3)
│   ├── Gamma近似Shadowed-Rician衰落CDF/PDF
│   ├── 中断概率: Leader (Theorem 1), Cluster (Theorem 2), 上下界 (Corollary 1)
│   └── 平均速率: Leader (Theorem 3), Cluster上下界 (Corollary 2)
├── monte_carlo.py               # Monte Carlo仿真验证模块
│   ├── 逆变换采样接触角、球冠偏角、方位角
│   ├── MC中断概率仿真（Leader/Cluster）
│   └── MC平均速率仿真（Leader/Cluster/Non-follower）
├── simulation.py                # 主仿真逻辑
│   ├── Fig. 2: 中断概率 vs SNR阈值
│   ├── Fig. 4: 平均速率 vs Follower数量和发射功率
│   └── Fig. 6: Leader-Follower vs Non-Follower速率对比
├── plotting.py                  # IEEE期刊风格绘图模块
│   └── 统一风格: Times New Roman, 三重区分(color+marker+linestyle)
├── run_reproduction.py          # 一键运行脚本
├── output/                      # 输出结果目录
│   ├── fig01_satellite_cluster_architecture.png  # 场景图（PDF提取）
│   ├── fig09_geometry_outside_spherical_cap.png  # 几何示意图（PDF提取）
│   ├── fig10_geometry_inside_spherical_cap.png   # 几何示意图（PDF提取）
│   ├── fig2_outage_probability.png               # 复现 Fig. 2
│   ├── fig4_avg_rate_vs_nf.png                   # 复现 Fig. 4
│   └── fig6_lf_vs_nf.png                        # 复现 Fig. 6
└── README.md
```

## 快速开始

```bash
cd "[032-TWC-2026]Coverage_and_Rate_Analysis_of_Follower-Based_LEO_Satellite_Networks_A_Stochastic_Geometry_Approach"
python run_reproduction.py
```

运行完成后，所有图表保存在 `output/` 目录下。

## 复现结果

### Fig. 2: 中断概率验证 (Analytical + Monte Carlo)

| SNR阈值 (dB) | Leader (Thm.1) | Cluster (Thm.2) | Upper Bound | Lower Bound |
|:---:|:---:|:---:|:---:|:---:|
| -10 | 0.086 | 0.004 | 0.060 | 0.120 |
| -5 | 0.515 | 0.398 | 0.415 | 0.607 |
| 0 | 0.974 | 0.974 | 0.892 | 0.987 |

MC仿真与解析结果高度吻合。

### Fig. 4: 平均速率 vs Follower数量

| 发射功率 | N_F=0 | N_F=10 | N_F=20 |
|:---:|:---:|:---:|:---:|
| 5 dBW | 0.107 | 0.111 | 0.114 |
| 10 dBW | 0.107 | 0.125 | 0.142 |
| 15 dBW | 0.107 | 0.165 | 0.224 |
| 20 dBW | 0.107 | 0.257 | 0.401 |

### Fig. 6: L.F vs N.F 速率对比

| 总功率 (dBW) | N.F速率 | L.F最高速率 | 增益 |
|:---:|:---:|:---:|:---:|
| 10 | 0.013 | 0.135 | +953% |
| 15 | 0.039 | 0.159 | +313% |
| 20 | 0.107 | 0.224 | +110% |
| 25 | 0.256 | 0.365 | +43% |
| 30 | 0.507 | 0.605 | +19% |

## 核心算法

1. **球面BPP模型**: N_L颗Leader卫星在轨道球面上均匀分布，最近卫星接触角服从 Lemma 1 的PDF
2. **Gamma近似**: 将Shadowed-Rician衰落用Gamma分布近似（式8-10），参数通过矩匹配获得
3. **中断概率**: 对接触角PDF积分，得到Leader（式16-17）和Cluster（式18-21）的闭合/半闭合表达式
4. **平均速率**: 通过四重数值积分（theta, psi, phi, w）计算Follower的速率贡献，Cluster速率为Leader速率加N_F个Follower贡献之和
5. **上下界**: 用球冠边界最近/最远距离替换Follower-user距离，得到紧致上下界

## 依赖库

```bash
pip install numpy matplotlib scipy
```

- **numpy**: 数值计算
- **scipy**: 数值积分 (`integrate.quad`), 特殊函数 (`gammainc`), 统计分布
- **matplotlib**: IEEE风格绑图
- **Pillow**: （可选）封面图生成
- **PyMuPDF (fitz)**: （可选）PDF渲染提取封面
