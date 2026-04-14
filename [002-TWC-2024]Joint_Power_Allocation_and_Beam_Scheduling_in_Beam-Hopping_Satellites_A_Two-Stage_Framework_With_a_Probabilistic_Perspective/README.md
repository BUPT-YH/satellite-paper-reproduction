# TWC-2024 论文复现代码

## 论文信息

**标题**: Joint Power Allocation and Beam Scheduling in Beam-Hopping Satellites: A Two-Stage Framework With a Probabilistic Perspective

**作者**: Lin Chen, Linlong Wu, Eva Lagunas, Anyue Wang, Lei Lei, Symeon Chatzinotas, Björn Ottersten

**发表**: IEEE Transactions on Wireless Communications, Vol. 23, No. 10, October 2024

## 项目结构

```
[2-TWC-2024]/
├── config.py                        # 仿真参数配置
├── inverse_matrix_optimization.py   # 阶段1: 逆矩阵优化算法 (Algorithm 1)
├── beam_scheduling.py               # 阶段2: 离散化 + 照明模式设计
├── simulation.py                    # 仿真实验模块
├── plotting.py                      # 绘图模块
├── run_reproduction.py              # 一键复现脚本
├── output/                          # 输出图表
│   ├── figure4_convergence.png
│   └── figure11_performance_comparison.png
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[2-TWC-2024]"
python run_reproduction.py
```

运行时间约 10 分钟。

## 复现结果

### Figure 4: 收敛性分析

Algorithm 1 在不同需求密度下的收敛曲线。算法在 2-5 次迭代内收敛，与论文报告一致。

### Figure 11: 性能对比

提出方法 vs 基线方法的能耗和需求匹配性能对比：

| 密度 | 提出方法 Jain 指数 | 基线 Jain 指数 | 能耗比 (提出/基线) |
|------|---------------------|----------------|---------------------|
| r=0.1 | 0.41 | 0.15 | 0.014 |
| r=0.3 | 0.82 | 0.26 | 0.027 |
| r=0.5 | 0.93 | 0.39 | 0.056 |

### 定性结论验证

| 论文结论 | 复现结果 |
|----------|----------|
| 提出方法能耗显著低于基线 | ✓ 验证通过 |
| Jain 公平性指数随密度增加 | ✓ 验证通过 |
| 提出方法公平性优于基线 | ✓ 验证通过 |
| 算法 2-3 次迭代收敛 | ✓ 验证通过 |

### 偏差说明

绝对数值与论文存在差异，原因：
1. 信道模型采用归一化简化模型，非论文的精确 GEO 卫星辐射模型
2. 基线方法实现为简化版贪心调度，非论文的完整基线
3. MPMM 求解器使用贪心近似，非论文的完整二元二次规划求解器

## 核心算法

### Algorithm 1: 逆矩阵优化
- 基于平均场理论解耦功率 p 和波束激活概率 ρ
- 通过 SCA 迭代求解凸优化问题 P4
- 利用 (I - GA)^{-1} 的一对一映射关系

### Algorithm 2: 离散化舍入
- 将连续概率转换为离散时隙需求
- 保持 Σd̂ = MK 约束

### 照明模式设计
- 二元二次规划 (BQP)
- 考虑干扰惩罚和切换延迟

## 依赖库

```bash
pip install numpy matplotlib
```

