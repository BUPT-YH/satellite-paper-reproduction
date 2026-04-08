# TWC-2025 论文复现代码

## 论文信息

**标题**: Feasibility Analysis of In-Band Coexistence in Dense LEO Satellite Communication Systems

**作者**: Eunsun Kim, Ian P. Roberts, Jeffrey G. Andrews

**发表**: IEEE Transactions on Wireless Communications, VOL. 24, NO. 2, February 2025

## 项目结构

```
├── config.py              # 仿真参数配置 (星座、天线、功率)
├── constellation.py       # Walker-Delta 星座生成与轨道传播
├── antenna.py             # UPA 相控阵天线方向图
├── channel.py             # 路径损耗、SNR、INR 计算
├── simulation.py          # 主仿真: 卫星选择策略 + 干扰分析
├── plotting.py            # 绘图模块 (Fig. 2-14)
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表 (12 张)
└── README.md
```

## 快速开始

```bash
cd "文章复现/[6-TWC-2025]Feasibility_Analysis_of_In-Band_Coexistence_in_Dense_LEO_Satellite_Communication_Systems"
python run_reproduction.py
```

## 复现结果

### Fig. 2: 波束方向图
64×64 星载天线 + 8×8/16×16/32×32 用户天线的归一化增益

### Fig. 3: 频谱效率损失
干扰 (INR) 对不同 SNR 下链路频谱效率的影响

### Fig. 4: 干扰上下界 CDF
绝对和条件干扰界的经验 CDF。结果与论文 Takeaway 1 一致:
- 几乎总存在可造成极高干扰的卫星对
- 也几乎总存在可将干扰降到极低的卫星对

### Fig. 5: 可行次级卫星数量
满足 INR 保护约束的 Kuiper 卫星数量。中位数 12-18 颗，与论文吻合

### Fig. 6-7: 贪心选择 SINR 和干扰
Max-SINR 优于 Max-SNR (Takeaway 3)，但不能保证主用户保护

### Fig. 8: 保护性选择 SINR
保护性 Max-SINR 在满足约束的同时次系统损失很小 (Takeaway 4)

### Fig. 9: 有用卫星数量
Δ=3dB 时 4-12 颗卫星可用，与论文一致

### Fig. 10-11: 角间距分析
约束越严格，次级卫星选择离贪心选择越远 (Takeaway 6)

### Fig. 13-14: 不确定性分析
即使在 γ=50° 不确定性下，仍有 ≥3 颗可用卫星 (Takeaway 7-8)

## 核心算法

1. **Walker-Delta 星座**: 基于 FCC 填报参数 (Starlink 4408 + Kuiper 3236)
2. **UPA 天线方向图**: 统一平面阵列的精确阵列因子计算
3. **卫星选择策略**: 贪心 Max-SNR/SINR, 保护性 Max-SNR/SINR, 不确定性下保障 SINR

## 仿真简化假设

- 时间分辨率: 2 分钟 (论文 30 秒)，每 3 步采样
- 采样城市: 6 个代表性城市 (论文 ~3000)
- 不确定性分析: 间隔采样以加速
- 共址用户假设 (论文最坏场景)

## 依赖库

```bash
pip install numpy matplotlib
```
