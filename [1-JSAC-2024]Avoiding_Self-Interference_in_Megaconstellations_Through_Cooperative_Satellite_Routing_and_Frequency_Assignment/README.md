# JSAC-2024 论文复现代码

## 论文信息

**标题**: Avoiding Self-Interference in Megaconstellations Through Cooperative Satellite Routing and Frequency Assignment

**作者**: Pachler N, Crawley E F, Cameron B G

**发表**: IEEE Journal on Selected Areas in Communications, Vol. 42, No. 11, November 2024

## 项目结构

```
[1-JSAC-2024]/
├── constellation_config.py   # 星座配置参数 (O3b mPower, Telesat, Starlink)
├── satellite_routing.py      # 卫星路由优化算法
├── frequency_assignment.py   # 频率分配优化算法
├── simulation.py             # 仿真实验模块
├── plotting.py               # 绘图模块
├── run_reproduction.py       # 一键复现脚本
├── output/                   # 输出图表
│   ├── figure5_convergence.png
│   ├── figure6_constellation.png
│   ├── figure7_performance.png
│   └── figure8_operations.png
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[1-JSAC-2024]"
python run_reproduction.py
```

## 复现结果

### Figure 5: 收敛性分析

干扰集合(R_E)随迭代逐渐收敛，收敛后吞吐量趋于稳定，约100次迭代后达到收敛。

### Figure 6: 星座覆盖可视化

伊比利亚半岛上空的波束覆盖和卫星位置分布。

### Figure 7: 性能对比

四种方法（ME-WF, IO-WF, ME-IO, IO-IO）的吞吐量和功耗对比：

| 星座 | 方法 | 吞吐量 (Gbps) | 功耗 (W) |
|------|------|---------------|----------|
| O3b mPower | ME-WF | 42.83 | 185.0 |
|  | IO-IO | 55.90 | 123.9 |
| Telesat Lightspeed | ME-WF | 87.53 | 215.7 |
|  | IO-IO | 113.86 | 151.8 |
| SpaceX Starlink | ME-WF | 124.41 | 290.9 |
|  | IO-IO | 160.13 | 201.9 |

### Figure 8: 运行仿真

运行期间吞吐量、功耗随时间变化，及波束-卫星切换事件。

### 定性结论验证

| 论文结论 | 复现结果 |
|----------|----------|
| IO-IO方法性能最优 | ✓ 验证通过 |
| 吞吐量提升约30% | ✓ 验证通过 |
| 功耗降低约30% | ✓ 验证通过 |
| 算法有限迭代内收敛 | ✓ 验证通过 |

### 偏差说明

绝对数值与论文存在差异，原因：
1. 星座参数部分来自公开资料推断
2. 仿真采用简化轨道模型
3. 频率分配使用启发式近似

## 核心算法

### 协作优化框架
- 迭代优化：卫星路由 + 频率分配交替求解
- 路由优化：最小化违反角分离条件的波束对
- 频率分配：确保频谱不重叠或极化方式不同

### 三种干扰避免条件
1. 波束对使用不重叠的频谱
2. 波束对使用不同的极化方式
3. 波束对由具有足够角分离的卫星服务

## 依赖库

```bash
pip install numpy matplotlib scipy scikit-learn
```

