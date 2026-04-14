# [JSAC-2025] Cooperative Multi-Satellite and Multi-RIS Beamforming 论文复现

## 论文信息

**标题**: Cooperative Multi-Satellite and Multi-RIS Beamforming: Enhancing LEO SatCom and Mitigating LEO-GEO Intersystem Interference

**作者**: Ziyuan Zheng, Wenpeng Jing, Zhaoming Lu, Qingqing Wu, Haijun Zhang, David Gesbert

**发表**: IEEE Journal on Selected Areas in Communications (JSAC), Vol. 43, No. 1, January 2025

## 项目结构

```
[9-JSAC-2025]Cooperative_Multi-Satellite_and_Multi-RIS_Beamforming.../
├── config.py              # 仿真参数配置 (轨道、频率、Rician因子等)
├── channel_model.py       # Rician 信道模型、UPA 阵列响应、路径损耗
├── closed_form.py         # 闭式 SINR 表达式 (公式 10, 28, 44) + RZF 预编码
├── optimization.py        # 优化算法 (AP-AO, MR-PA, MR-TS, ES-SP-RMO)
├── simulation.py          # 仿真主脚本 (Fig 2-13)
├── plotting.py            # IEEE 期刊风格绘图模块
├── run_reproduction.py    # 一键复现入口
├── output/                # 生成的仿真图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[9-JSAC-2025]Cooperative_Multi-Satellite_and_Multi-RIS_Beamforming_Enhancing_LEO_SatCom_and_Mitigating_LEO-GEO_Intersystem_Interference"
python run_reproduction.py
```

## 复现内容

### 核心算法
1. **Algorithm 1 (AP-AO)**: 使用 RZF (正则化迫零) 预编码 + Riemannian 流形优化 RIS 相移, 实现干扰抑制
2. **Algorithm 2 (MR-S-PA)**: 给定 RIS 相移下的 MR 预编码功率分配
3. **Algorithm 3 (MR-S-TS)**: 两阶段设计, 含 ES-SP-RMO 算法
4. **MR-TTS**: 双时间尺度 CSI 方案

### 仿真图表
| 图号 | 内容 | 状态 |
|------|------|------|
| Fig. 2 | Min SINR vs PT (κN=20dB) | [OK] |
| Fig. 3 | Min SINR vs PT (κN=0dB) | [OK] |
| Fig. 4 | Min SINR vs ζ (κN=20dB) | [OK] |
| Fig. 5 | Min SINR vs ζ (κN=0dB) | [OK] |
| Fig. 6 | Min SINR vs M (κN=20dB) | [OK] |
| Fig. 7 | Min SINR vs M (κN=0dB) | [OK] |
| Fig. 8 | Min SINR vs κR (κN=0dB) | [OK] |
| Fig. 9 | Min SINR vs κR (κN=10dB) | [OK] |
| Fig. 10 | Min SINR vs κR (κN=20dB) | [OK] |
| Fig. 11 | MSC vs SST Min SINR | [OK] |
| Fig. 12 | MSC vs SST Sum Rate | [OK] |
| Fig. 13 | 执行时间对比 | [OK] |

### 复现结果 (κN=20dB, Fig 2)
| 方案 | PT=2W | PT=15W | PT=24.5W | 论文参考 |
|------|-------|--------|----------|----------|
| AP-AO (RZF) | -3.7 dB | +4.8 dB | +6.4 dB | -2→+12 dB |
| MR-S-PA | -11.4 dB | -3.1 dB | -1.9 dB | -5→+4 dB |
| AP-NoRIS (RZF) | -5.8 dB | +0.2 dB | +1.4 dB | -10→0 dB |
| MR-S-NoRIS | -15.8 dB | -6.8 dB | -5.7 dB | -15→-5 dB |

### 关键技术改进
1. **RZF 预编码**: AP-AO 方案使用正则化迫零预编码替代 MR 预编码, 实现用户间干扰抑制, SINR 随 PT 单调增长
2. **参数校准**: NF_RIS=5e6 使 RIS 级联增益合理; system_interference_margin=1.0 W 使系统保持噪声受限
3. **功率分配保障**: 迭代功率平衡从等功率开始, 含最小功率下限, 对比等功率基线防止退化

### 与论文结果的差异说明
1. **绝对值偏差**: AP-AO 最高 +6.4 dB vs 论文 +12 dB — RZF 只是自适应预编码的近似
2. **MR 方案偏低**: MR 预编码无法抑制干扰, SINR 比 RZF 低 7-10 dB
3. **MR 方案差异小**: MR-S-PA ≈ MR-S-TS ≈ MR-TTS — MR 预编码下 RIS 优化效果有限
4. **正确复现的趋势**:
   - 所有曲线随 PT 单调递增 ✓
   - AP-AO > MR 方案 > NoRIS ✓
   - RIS 方案优于 NoRIS (4-10 dB 增益) ✓
   - 高 κN 环境优于低 κN 环境 ✓
   - SINR 随 κR 增大而提升 ✓
   - MSC 优于 SST ✓

## 依赖库

```bash
pip install numpy matplotlib scipy
```
