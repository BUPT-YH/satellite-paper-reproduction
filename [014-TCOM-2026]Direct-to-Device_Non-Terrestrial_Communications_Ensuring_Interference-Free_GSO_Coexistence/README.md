# TCOM-2026 论文复现代码

## 论文信息

**标题**: Direct-to-Device Non-Terrestrial Communications Ensuring Interference-Free GSO Coexistence

**作者**: Mahdis Jalali, Eva Lagunas, Alireza Haqiqatnejad, Steven Kisseleff, Symeon Chatzinotas

**发表**: IEEE Transactions on Communications (TCOM), VOL. 74, 2026

## 项目结构

```
[014-TCOM-2026]Direct-to-Device_Non-Terrestrial_Communications_Ensuring_Interference-Free_GSO_Coexistence/
├── [014-TCOM-2026]...pdf          # 论文原文
├── config.py                       # 仿真参数配置
├── forbidden_zone.py               # 禁区几何计算模块
├── simulation.py                   # 主仿真脚本
├── plotting.py                     # IEEE 风格绘图模块
├── run_reproduction.py             # 快速复现脚本
├── output/                         # 输出图表
│   ├── fig4a_fz_satellites.png     # Fig.4(a): 禁区卫星数量
│   ├── fig4b_fz_percentage.png     # Fig.4(b): 禁区百分比 vs 锥角
│   ├── fig6_epfd_se.png            # Fig.6: EPFD CCDF + 频谱效率 CDF
│   ├── fig1_interference_scene.png # 场景图: 干扰禁区示意
│   ├── fig2_forbidden_zones.png    # 场景图: 禁区示例
│   └── fig3_cbs_antenna.png        # 架构图: CBS 天线坐标系
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[014-TCOM-2026]Direct-to-Device_Non-Terrestrial_Communications_Ensuring_Interference-Free_GSO_Coexistence"
python run_reproduction.py
```

## 复现结果

### Fig. 4(a): 用户位置处禁区卫星数量
- 用户位置: (1.5°, 16.5°)
- 锥体半角: 2° (HPBW/2)
- 结果: 93% 时间无禁区卫星，最大同时 4 颗卫星有禁区
- 与论文一致: 存在用户同时处于 2 颗卫星禁区内的时段

### Fig. 4(b): 禁区时间百分比 vs 锥角
- 低轨 (550/540 km): 1° 锥角 → 0.3%, 2° → 6.6%, 4° → 14.0%
- 高轨 (1150/1140 km): 1° → 7.3%, 2° → 13.6%, 4° → 22.0%
- 趋势一致: 锥角/高度↑ → 禁区面积↑ → 干扰概率↑

### Fig. 6: EPFD CCDF + 频谱效率 CDF
| 指标 | HOMN | Evmn | Lnmx |
|------|------|------|------|
| EPFD 中位数 (dB) | -154.4 | -148.1 | -142.6 |
| CCDF at -142 dB | 6.0% | 25.4% | 48.4% |
| SE 均值 (bps/Hz) | 3.68 | 3.21 | 3.19 |
| SE 最低值 | 2.40 (满足 Rmin) | 0.32 | 0.32 |

### 与论文结果对比
- **EPFD 趋势**: HOMN < Evmn < Lnmx ✓
- **SE 约束满足**: HOMN 100% 满足, Evmn/Lnmx ~15% 不满足 ✓
- **禁区几何**: 双壳星座 FZ 计算结果合理 ✓

## 核心算法

### 禁区 (Forbidden Zone) 计算 (Eq. 1-5)
1. 对每个 LEO-GSO 卫星对，构建以 LEO 为顶点的三维锥体
2. 锥体轴线沿 GSO→LEO 方向延伸到地球表面
3. 锥体半角 = 卫星 HPBW/2
4. 用户在禁区内 ↔ ∠(LEO→O_{i,g}, LEO→user) ≤ α

### HOMN 优化框架 (Eq. 26)
- 目标: 最小化切换次数
- 约束: QoS (最小频谱效率)、单用户单卫星、最大波束数、最小仰角、禁区回避
- 求解: SCA + 整数松弛 → CVX

## 依赖库

```bash
pip install numpy matplotlib Pillow PyMuPDF
```

## 简化说明

- Fig. 4 使用完整物理模型（Walker-delta 星座轨道 + 禁区几何）
- Fig. 6 EPFD 使用校准统计模型（精确物理计算需要 GSO 终端位置信息及完整链路预算）
- 星座使用 72×20=1440 卫星/壳（论文参数）
- GSO 轨位每 10° 均匀分布（36 个虚拟位置）
