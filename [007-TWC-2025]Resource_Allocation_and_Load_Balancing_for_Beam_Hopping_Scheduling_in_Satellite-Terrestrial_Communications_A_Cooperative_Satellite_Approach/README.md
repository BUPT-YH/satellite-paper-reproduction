# TWC-2025 论文复现代码

## 论文信息

**标题**: Resource Allocation and Load Balancing for Beam Hopping Scheduling in Satellite-Terrestrial Communications: A Cooperative Satellite Approach

**作者**: Guanhua Wang, Fang Yang, Jian Song, Zhu Han

**发表**: IEEE Transactions on Wireless Communications, Vol. 24, No. 2, February 2025

## 项目结构

```
[7-TWC-2025]Resource_Allocation_and_Load_Balancing.../
├── config.py              # 仿真参数配置
├── beam_hopping_drl.py    # DRL跳波束调度 (Double DQN)
├── resource_allocation.py # MM算法资源分配
├── load_balancing.py      # ISL负载均衡
├── simulation.py          # 主仿真逻辑
├── plotting.py            # 绘图模块
├── run_reproduction.py    # 快速复现脚本
├── output/                # 输出图表
│   ├── fig3_training_reward.png      # DQN训练收敛曲线
│   ├── fig4_mm_convergence.png       # MM算法收敛性能
│   ├── fig5_method_comparison.png    # 不同方法对比
│   ├── fig6_beta_tradeoff.png        # β折衷系数分析
│   ├── fig7_traffic_load.png         # 消融实验: 输入流量
│   └── fig8_satellite_number.png     # 不同卫星数量
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[7-TWC-2025]Resource_Allocation_and_Load_Balancing_for_Beam_Hopping_Scheduling_in_Satellite-Terrestrial_Communications_A_Cooperative_Satellite_Approach"
python run_reproduction.py
```

## 复现结果

### Fig.3: DQN训练收敛曲线
训练过程中平均奖励逐渐提升并稳定，验证了DRL方法的收敛性。

### Fig.4: MM算法收敛性能
全星座优化、邻近卫星协作和无协作三种策略均快速收敛。邻近卫星协作性能损失<0.8%。

### Fig.5: 不同方法对比
提出方法在不同输入流量下均优于预调度、相邻波束避免和最大USWG方法。

### Fig.6: 折衷系数β分析
β增大时吞吐量提升但延迟度量增加，体现了吞吐量-延迟的权衡关系。

### Fig.7: 消融实验
DRL跳波束调度对性能提升最显著，资源分配和负载均衡也有明显贡献。

### Fig.8: 不同卫星数量
提出方法在不同星座规模下均保持优越性能，负载均衡在大型星座中尤为重要。

## 核心算法

1. **DRL跳波束调度**: 多智能体Double DQN，Q值分解为各小区贡献之和，ε-greedy策略
2. **MM资源分配**: Majorization-Minimization算法迭代求解非凸资源分配，松弛为线性规划
3. **负载均衡**: 区分充足/过载场景，二次变换+混合块逐次近似算法求解

## 依赖库

```bash
pip install numpy matplotlib torch scipy
```
