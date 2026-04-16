# JSAC-2026 论文复现代码

## 论文信息

**标题**: LLM-Aided Spectrum-Sharing LEO Satellite Communications

**作者**: Zihan Ni, Zizheng Hua, Xuanhe Yang, Rui Zhang, Shuai Wang, Gaofeng Pan

**发表**: IEEE Journal on Selected Areas in Communications (JSAC), Vol. 44, 2026

## 项目结构

```
[10-JSAC-2026]LLM-Aided_Spectrum-Sharing_LEO_Satellite_Communications/
├── config.py              # 仿真参数配置
├── channel_model.py       # Shadowed Rician 信道模型 + 卫星位置分布
├── outage_probability.py  # 中断概率解析公式 (Pout,1, Pout,2) + 蒙特卡洛仿真
├── resource_allocation.py # 三种传输方案 (顺序/混合/LLM决策)
├── plotting.py            # IEEE 期刊风格绘图
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[10-JSAC-2026]LLM-Aided_Spectrum-Sharing_LEO_Satellite_Communications"
python run_reproduction.py
```

## 复现结果

### Fig. 1: 系统模型 (从论文提取)
### Fig. 2: 轴截面图 (从论文提取)
### Fig. 3: 七区域 CDF 推导 (从论文提取)
### Fig. 4: LLM 辅助系统架构 (从论文提取)

### Fig. 8: Pout,1 vs PS (频谱共享传输)
频谱共享传输（LLM决策后利用频谱空洞）的中断概率，随发射功率变化。
- 解析曲线与蒙特卡洛仿真完全吻合
- dSD 越大，OP 越高（路径损耗增大）

| dSD (km) | Pout,1 (PS=5dBW) | Pout,1 (PS=30dBW) |
|----------|------------------|--------------------|
| 600      | 2.12e-01         | 4.26e-04           |
| 800      | 4.25e-01         | 7.82e-04           |
| 1000     | 6.60e-01         | 1.25e-03           |

### Fig. 9: Pout,2 vs γth (固定频段传输)
固定频段传输的中断概率，含硬核 Poisson 点过程干扰建模。

| dSD (km) | Pout,2 (γth=-5dB) | Pout,2 (γth=20dB) |
|----------|-------------------|--------------------|
| 600      | 3.72e-02          | 9.90e-01           |
| 800      | 8.06e-02          | 9.93e-01           |
| 1000     | 1.46e-01          | 9.94e-01           |

### Fig. 10: Pout,2 vs γth (不同 λe·Δfs)
干扰密度越大，中断概率越高。

### Fig. 11: LLM决策 vs 固定频段 OP 对比
- **LLM决策** (Pout,1): 显著低于固定频段
- PS=20dBW 时：LLM OP ≈ 10^{-3}，固定频段 OP ≈ 10^{-1}，提升约两个数量级

### Fig. 12: 三种方案传输数据量
- 顺序传输 (64QAM): 33.35s 完成全部 54.02 GB
- 混合传输 (BPSK+16QAM): 83.41s
- LLM决策 (DeepSeek r1): 40.00s，接近最优

### Fig. 13: 传输等待时延对比
- LLM 决策方案：所有业务同时传输，等待时延为 0
- 顺序/混合方案：后发业务需等待前面业务完成

## 核心算法

1. **卫星位置分布**: 基于随机几何推导 (r,θ) 联合 CDF (7区域分段)，PDF = 2π/V · r² · sin(θ)
2. **Shadowed Rician 信道**: 平均阴影衰落 (mn=5, Ω=0.835, b=0.126)
3. **Pout,1 (Eq.40)**: 频谱共享无干扰 OP = Σ ζ(k)/(β-δ)^{k+1} · γ(k+1, (β-δ)·σ²n·d^α·γth/PS)
4. **Pout,2 (Eq.45-46)**: 固定频段含干扰 OP，Gaussian-Chebyshev 数值积分 + Alzer 近似
5. **LLM 资源分配**: Prompt Engineering 驱动的多业务动态调制与带宽分配

## 仿真参数假设

论文未明确给出以下参数，本复现采用合理假设：
- **mn = 5**: Shadowed Rician 平均阴影衰落
- **Θ0 = 25°**: 天线波束半角
- **PE/PS = 0.1**: 干扰功率与信号功率比
- **γth = 10 dB**: Fig.8 中固定的 SINR 门限

## 依赖库

```bash
pip install numpy matplotlib scipy
```
