# TCOM-2026 论文复现代码

## 论文信息

**标题**: Co-Existence Analysis of Terrestrial and Non-Terrestrial Networks in S-Band Using Stochastic Geometry

**作者**: Niloofar Okati, Andre Noll Barreto, Luis Uzeda Garcia, Jeroen Wigard

**发表**: IEEE Transactions on Communications, Vol. 74, pp. 4431-4445, 2026

## 项目结构

```
[034-TCOM-2026]Co-Existence_Analysis_of_Terrestrial_and_Non-Terrestrial_Networks_in_S-Band_Using_Stochastic_Geometry/
├── config.py              # 仿真参数配置（星座、频段、功率、天线增益等）
├── stochastic_geometry.py # 核心随机几何计算模块（CDF/PDF/Laplace变换/覆盖概率）
├── plotting.py            # IEEE期刊风格绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
│   ├── fig01_coexistence_scenarios.png  # 场景图：两种共存场景
│   ├── fig02_case2_geometry.png         # 场景图：Case II几何模型
│   ├── fig5_case1_coverage_fullload.png   # 图5(a): Case I覆盖概率, 100%负载
│   ├── fig5_case1_coverage_25pctload.png  # 图5(b): Case I覆盖概率, 25%负载
│   ├── fig8_case2_coverage_fullload.png   # 图8(a): Case II覆盖概率, 100%负载
│   ├── fig8_case2_coverage_25pctload.png  # 图8(b): Case II覆盖概率, 25%负载
│   └── fig11_case1_vs_case2.png           # 图11: Case I vs Case II对比
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[034-TCOM-2026]Co-Existence_Analysis_of_Terrestrial_and_Non-Terrestrial_Networks_in_S-Band_Using_Stochastic_Geometry"
python run_reproduction.py
```

运行时间约 20 分钟（Intel i7，含 Monte Carlo 验证）。

## 复现结果

### SINR阈值范围

所有覆盖概率曲线在 T ∈ [-50, 30] dB 范围内绘制（0.5 dB 步长，161个点），确保曲线从 P_c ≈ 1 开始。

### 图5: Case I 覆盖概率 vs SINR阈值

NTN下行干扰TN下行场景，对比城市/农村 x 三种卫星高度(200/600/1200km) x 100%/25%负载。

关键发现：
- 城市场景中NTN干扰影响较小（TN干扰占主导），三种卫星高度的曲线几乎重合
- 农村场景中NTN干扰影响显著，低轨卫星(200km)干扰最强
- BPP模型具有尺度不变性：无NTN干扰时城市和农村覆盖概率完全相同

典型数值（T=0dB, 100%负载）：
- Urban基线（无NTN干扰）: P_c = 0.44
- Urban各高度（200/600/1200km）: P_c均约0.44（几乎无影响）
- Rural基线: P_c = 0.44
- Rural 200km: P_c = 0.09
- Rural 600km: P_c = 0.25
- Rural 1200km: P_c = 0.36

### 图8: Case II 覆盖概率 vs SINR阈值

NTN上行干扰TN下行场景，卫星高度固定600km，对比不同NTN UE数量 $N_u \in \{100, 1000, 2000\}$ 和隔离距离 $r_{iso} \in \{0, 2 \times d_{ISD}\}$ 的组合。

Case II NTN UE分布模型：
- NTN UE分布在环形区域 $[r_{TN} + r_{iso}, R_{NTN}]$ 内，其中 $R_{NTN} = 25$ km为卫星波束覆盖半径
- 路径损耗指数 $\alpha_{NTN} = 2$（自由空间传播）

关键发现：
- Urban场景 $r_{iso}=0$: $N_u=100$ 时 P_c(0dB)=0.21，$N_u=2000$ 时 P_c(0dB)=0.04
- Rural场景 $r_{iso}=0$: $N_u=100$ 时 P_c(0dB)=0.008，$N_u=2000$ 时 P_c(0dB)=0.001
- Rural场景 $r_{iso}=2\times d_{ISD}$: 内径超出波束覆盖半径，无NTN UE存在，等价于无NTN干扰基线
- $N_u$增大使干扰显著增加，隔离距离是控制Case II干扰的关键参数

注：Case II Rural $r_{iso}=0$ 场景中，大量NTN UE（每个EIRP=200W）在[16, 25]km环形区域产生极强干扰，$N_u \geq 1000$时曲线无法在常规SINR范围内达到P_c=1。这是该极端干扰场景的物理本质。

### 图11: Case I vs Case II 对比

单图包含8条曲线的对比：
- 4条Case II曲线（600km, Urban场景）：$N_u \in \{100, 2000\}$ x $r_{iso} \in \{0, 2\times d_{ISD}\}$
- 4条Case I曲线：Urban/Rural x 200km/1200km
- 加1条No Spectrum Sharing基线

直接对比两种共存场景的覆盖概率，提供动态频谱共享策略指导。

### Monte Carlo验证

Monte Carlo仿真验证了解析结果的正确性（50000次试验）：

| 场景 | 解析值 | MC仿真值 | 偏差 |
|------|--------|----------|------|
| Urban, 600km, T=0dB, 100%load | 0.4419 | 0.4368 | 0.51% |
| Urban, 600km, T=10dB, 100%load | 0.1084 | 0.1099 | 0.15% |
| Rural, 600km, T=0dB, 100%load | 0.2487 | 0.2477 | 0.10% |

## 核心算法

论文基于binomial point process (BPP)建模TN BS分布，利用Nakagami-m衰落信道的Laplace变换，推导了两种共存场景下覆盖概率的精确解析表达式：

1. **服务距离分布**（公式3）：$F_{R_0}(r_0) = 1 - (1 - F_R(r_0))^{N_c}$
2. **TN干扰Laplace变换**（公式9）：基于条件距离分布的积分，采用CDF-based Riemann-Stieltjes数值积分避免PDF数值微分的误差放大
3. **NTN干扰Laplace变换**（公式10/15）：天顶简化版闭合形式（Case I），环形区域积分（Case II）
4. **覆盖概率**（公式8/14）：对服务距离积分，包含噪声项、TN干扰项和NTN干扰项

### 数值积分方法

TN干扰和NTN UL干扰的Laplace变换采用CDF-based Riemann-Stieltjes积分（替代PDF-based scipy.quad），使用非均匀网格（512个点），避免数值微分误差在指数运算（$L^{N_{active}}$ 或 $L^{N_u}$）中被放大的问题。

## 依赖库

```bash
pip install numpy matplotlib scipy
```
