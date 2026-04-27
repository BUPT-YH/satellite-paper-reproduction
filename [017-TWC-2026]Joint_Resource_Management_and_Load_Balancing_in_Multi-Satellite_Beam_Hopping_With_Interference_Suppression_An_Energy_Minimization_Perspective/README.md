# [TWC-2026] 论文复现代码

## 论文信息

**标题**: Joint Resource Management and Load Balancing in Multi-Satellite Beam Hopping With Interference Suppression: An Energy Minimization Perspective

**作者**: Guanhua Wang, Fang Yang, Jiaji Liu, Jian Song, Zhu Han

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[017-TWC-2026].../
├── config.py              # 仿真参数配置
├── channel_model.py       # 信道模型 (卫星网络拓扑 + 归一化信道 + 地面距离覆盖)
├── optimizer.py           # BCD + 统一 Lyapunov 功率分配 + QP 负载均衡
├── baselines.py           # 基线方法 (DRL, Pre-scheduling, No Freq Div, No LB, Max USWG)
├── simulation.py          # 时隙仿真框架
├── plotting.py            # IEEE 期刊风格绘图
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[017-TWC-2026]..."
python run_reproduction.py
```

## 复现结果

### Fig. 2: BCD 算法收敛性
BCD 算法在不同卫星数量(|S|=2,3,4)下的收敛曲线，均在6步内收敛。

### Fig. 3: V 权衡系数影响
不同 V 值下的平均队列长度和平均功率，展示 Lyapunov O(V)~O(1/V) 权衡。

| V  | 平均功率 (W) | 平均队列长度 |
|----|-------------|------------|
| 50 | 63.5        | 24.0       |
| 100| 55.9        | 23.5       |
| 200| 34.7        | 23.8       |
| 400| 22.1        | 28.2       |

### Fig. 4: 干扰阈值与 ISL 传输限制
不同 Zmax 和 cmax 下的平均功率（c_max=5为例）：

| Zmax (dBW) | 平均功率 (W) |
|------------|-------------|
| -140       | 5.3         |
| -130       | 8.0         |
| -125       | 14.8        |
| -120       | 33.7        |

### Fig. 6: 方法对比
20 Gbps 需求下各方法的平均功率：

| 方法         | 平均功率 (W) | 相对 Proposed |
|-------------|-------------|--------------|
| Proposed    | 34.5        | —            |
| DRL         | 38.8        | +12%         |
| Pre-sched   | 37.9        | +10%         |
| No Freq Div | 65.0        | +88%         |
| Without LB  | 37.7        | +9%          |
| Max USWG    | 55.7        | +61%         |

## 核心算法

1. **Lyapunov drift-plus-penalty**: 将长期能量最小化转化为逐时隙优化
2. **BCD**: 交替优化 BH 模式 F、功率分配 P、负载均衡 B
3. **统一 Lyapunov 功率分配**: 所有方法共享相同的功率计算逻辑，差异仅来自 BH 和 LB
4. **干扰约束**: 在功率分配中直接缩放违反 GSO 干扰阈值的波束

## 依赖库

```bash
pip install numpy matplotlib scipy cvxpy
```
