# TVT-2025 论文复现代码

## 论文信息

**标题**: Beam Footprint Design, Scheduling, and Spectrum Assignment in Low Earth Orbit Mega-Constellations

**发表**: IEEE Transactions on Vehicular Technology, 2025

## 项目结构

```
[3-TVT-2025]/
├── constellation_config.py    # 星座配置参数 (O3b mPower, Telesat, Starlink等)
├── beam_footprint_design.py   # 波束足迹设计模块
├── user_scheduling.py         # 用户调度优化模块
├── spectrum_assignment.py     # 频谱分配优化模块
├── simulation.py              # 主仿真脚本
├── plotting.py                # 绘图模块
├── run_reproduction.py        # 快速复现脚本 (推荐使用)
├── output/                    # 输出图表目录
│   ├── figure5_convergence.png
│   ├── figure6_beam_footprint.png
│   ├── figure7_performance.png
│   └── figure8_operations.png
└── README.md                  # 本文件
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[3-TVT-2025]"
python run_reproduction.py
```

## 生成的图表

### Figure 5: 收敛分析
展示优化算法的收敛过程，包括：
- R_E 干扰集合大小随时间的演变
- 系统吞吐量的增长

### Figure 6: 星座覆盖可视化
展示波束足迹设计和频率复用模式：
- 伊比利亚半岛上空的波束覆盖
- 卫星位置和仰角分布
- 四色频率复用模式

### Figure 7: 性能比较
比较四种优化方法：
- **ME-WF**: 最大仰角路由 + 注水频谱分配 (基线)
- **IO-WF**: 整数优化路由 + 注水频谱分配
- **ME-IO**: 最大仰角路由 + 整数优化频谱
- **IO-IO**: 联合优化 (最佳性能)

### Figure 8: 运行仿真
展示系统运行时的性能：
- 吞吐量随时间变化
- 功耗随时间变化
- 簇-卫星切换事件

## Table II: 性能对比结果

| 星座 | 方法 | 吞吐量 (Gbps) | 功耗 (W) | 频谱 (MHz) | 活跃波束 |
|------|------|---------------|----------|------------|----------|
| O3b mPower | ME-WF | 74.28 | 1003 | 14473 | 100 |
|  | IO-WF | 81.65 | 879 | 14473 | 100 |
|  | ME-IO | 88.70 | 819 | 14473 | 100 |
|  | IO-IO | 101.09 | 717 | 14473 | 100 |
| Telesat Lightspeed | ME-WF | 71.20 | 503 | 13860 | 100 |
|  | IO-WF | 78.11 | 439 | 13860 | 100 |
|  | ME-IO | 84.94 | 409 | 13860 | 100 |
|  | IO-IO | 96.84 | 362 | 13860 | 100 |
| SpaceX Starlink | ME-WF | 41.30 | 203 | 7913 | 100 |
|  | IO-WF | 43.72 | 175 | 7913 | 100 |
|  | ME-IO | 48.46 | 163 | 7913 | 100 |
|  | IO-IO | 55.57 | 149 | 7913 | 100 |

## 核心算法

### 1. 波束足迹设计 (Section III)
- 基于用户分布的K-means聚类
- 最小化波束重叠区域
- 优化功率分配

### 2. 用户调度 (Section IV)
- 整数优化问题建模
- 干扰感知的时隙分配
- 迭代优化算法

### 3. 频谱分配 (Section V)
- 频率复用优化
- 极化复用
- 四色定理应用

## 依赖库

```bash
pip install numpy matplotlib scikit-learn
```

## 注意事项

1. `run_reproduction.py` 是快速复现脚本，生成简化版结果
2. `simulation.py` 包含完整仿真逻辑，运行时间较长
3. 输出图表保存在 `output/` 目录

## 作者

复现代码基于 TVT-2025 论文实现
