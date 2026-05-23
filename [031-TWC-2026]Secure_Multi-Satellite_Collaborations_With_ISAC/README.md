# TWC-2026 论文复现代码

## 论文信息

**标题**: Secure Multi-Satellite Collaborations With ISAC

**作者**: Xuyang Zhang, Zihan Ni, Xuanhe Yang, Xiaqing Miao, Shuai Wang, Gaofeng Pan, Jianping An, Dusit Niyato

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[031-TWC-2026]Secure_Multi-Satellite_Collaborations_With_ISAC/
├── config.py              # 仿真参数配置 (TABLE II)
├── isac_msc.py            # 核心算法模块 (信道模型、感知SNR、CRB)
├── simulation.py          # 蒙特卡洛仿真框架
├── plotting.py            # IEEE 期刊风格绘图
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
│   ├── fig01_ISAC_MSC_system_model.png
│   ├── fig02_LEO_satellite_visible_relationship.png
│   ├── Fig3_sensing_SNR_vs_Pm.png
│   ├── Fig6_sensing_SNR_vs_M0.png
│   └── Fig9_CRB_vs_Pm.png
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[031-TWC-2026]Secure_Multi-Satellite_Collaborations_With_ISAC"
python run_reproduction.py
```

## 复现结果

### Fig. 3: 感知 SNR vs 功率预算 P_m

- 9 种算法组合的 SNR 随 P_m (5-30 dBW) 单调递增
- DP-JSC-BF 最优 (P_m=25 dBW 时约 47 dB)，SHP-PA 最差 (约 34 dB)
- 算法排序: DP-JSC-BF > DP-IA > DP-PA > CP-JSC-BF > CP-IA > CP-PA > SHP-JSC-BF > SHP-IA > SHP-PA
- 最大差距约 12.7 dB，与原文趋势一致

### Fig. 6: 感知 SNR vs 协作卫星数 M_0

- SNR 随 M_0 (1-9) 近似线性增长，P_m=25 dBW
- M_0=1 到 M_0=6 提升约 20.7 dB，展示多星协作增益
- 算法排序与 Fig. 3 一致

### Fig. 9: CRB 定位误差 vs P_m (不同 M_0)

- CRB^{1/2} 随 P_m 增大而减小 (对数刻度近似线性)
- M_0=6, P_m=25 dBW 时 CRB 约 5 m
- M_0 从 1 到 9，CRB 从 ~30 m 降至 ~3.4 m

## 核心算法

论文提出交替迭代优化框架：
1. **卫星分配** (固定 BF): DP (离散PSO)、CP (连续PSO)、SHP (基准)
2. **波束赋形** (固定分配): PA (功率近似)、IA (内近似)、JSC-BF (SDP联合优化)
3. 交替迭代直到收敛

最优组合: DP-JSC-BF

## 依赖库

```bash
pip install numpy matplotlib scipy
```
