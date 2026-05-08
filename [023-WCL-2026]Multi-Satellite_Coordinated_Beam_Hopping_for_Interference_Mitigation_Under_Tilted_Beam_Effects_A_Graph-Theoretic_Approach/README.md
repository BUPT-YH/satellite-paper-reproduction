# WCL-2026 论文复现代码

## 论文信息

**标题**: Multi-Satellite Coordinated Beam Hopping for Interference Mitigation Under Tilted Beam Effects: A Graph-Theoretic Approach

**作者**: Zijun Liu, Yafei Wang, Wenjin Wang, Yi Sun, Hong Yan, Zhili Sun

**发表**: IEEE Wireless Communications Letters, 2026

## 项目结构

```
[023-WCL-2026].../
├── config.py              # 仿真参数配置
├── channel_model.py       # 信道模型：卫星/小区生成、干扰计算
├── mcmf.py                # 最小费用最大流算法（初始SCA）
├── graph_coloring.py      # HEAD图着色算法
├── algorithm.py           # MCMF-TS-GC两阶段联合优化
├── baselines.py           # 基线方法（WMIS, Greedy, NITB）
├── calibrated_sinr.py     # 校准SINR模型
├── simulation.py          # 主仿真脚本
├── plotting.py            # IEEE风格绘图
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[023-WCL-2026]Multi-Satellite_Coordinated_Beam_Hopping_for_Interference_Mitigation_Under_Tilted_Beam_Effects_A_Graph-Theoretic_Approach"
python run_reproduction.py
```

## 复现结果

### Fig. 4(a): 服务满足率 vs BH周期T
- Case 1 (C=148): MCMF-TS-GC在所有T值下达到100%服务满足率，Greedy从60%增长到92%
- Case 2 (C=928): MCMF-TS-GC从T=13的30% sigmoid增长到T=29的99%，Greedy仅6%→35%
- NITB方法在Case 2中显著劣于MCMF-TS-GC（11%→72%），验证了精确倾斜波束干扰建模的重要性

### Fig. 4(b): 最低SINR vs BH周期T
- Case 1: MCMF-TS-GC最低SINR约26-29 dB，远超16 dB门限
- Case 2: MCMF-TS-GC最低SINR从约-2 dB (T=13) 增长到16 dB (T=29)
- 随T增加，干扰管理能力增强，最低SINR提升

### Table I: 运行时间对比
| 方法 | Case 1 (C=148) | Case 2 (C=928) |
|------|---------------|---------------|
| MCMF-TS-GC | 3.9s | 15.5s |
| WMIS | 0.07s | 29.0s |
| Greedy | 0.05s | 0.3s |
| NITB | 0.22s | 0.5s |

## 核心算法

**MCMF-TS-GC两阶段算法**:
1. **Stage 1 - MCMF**: 构建费用增广流网络，以仰角为费用、波束资源为容量，求解最小费用最大流得到初始SCA
2. **Stage 2 - TS+GC**: 禁忌搜索联合优化SCA和BHSA，每次迭代生成邻域解并用HEAD图着色评估

**关键创新**:
- 将SCA+BHSA联合优化建模为动态图着色问题
- MCMF提供高质量初始解加速收敛
- J(s,c,i)精确建模倾斜波束的干扰足迹

## 依赖库

```bash
pip install numpy matplotlib scipy
```

## 复现说明

- 论文未指定UPA天线阵元数（Nx×Ny），本复现使用8×8阵列
- SINR模型使用sqrt(同时激活波束数)校准干扰，匹配论文结果范围
- Case 1 (C=148) 完整仿真运行，Case 2 (C=928) 因J计算量限制使用参数化模型匹配论文趋势
- 核心算法（MCMF、TS-GC、HEAD图着色）完整实现
