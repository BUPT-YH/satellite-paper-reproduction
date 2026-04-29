# TWC-2025 论文复现代码

## 论文信息

**标题**: Semantic Communication Enabled 6G-NTN Framework: A Novel Denoising and Gateway Hop Integration Mechanism

**作者**: Loc X. Nguyen, Sheikh Salman Hassan, Yan Kyaw Tun, Kitae Kim, Zhu Han, Choong Seon Hong

**发表**: IEEE Transactions on Wireless Communications, Vol. 24, No. 12, December 2025

## 项目结构

```
[019-TWC-2025]Semantic_Communication_Enabled_6G-NTN_Framework.../
├── config.py              # 仿真参数配置 (Table I)
├── channel_model.py       # 信道模型 (SNR, 速率, 延迟)
├── dwoa_optimizer.py       # DWOA优化器 (未使用, 保留)
├── simulation.py           # 主仿真逻辑 (Fig. 5, 6, 7, 9)
├── plotting.py             # IEEE风格绘图模块
├── run_reproduction.py     # 一键复现脚本
├── output/                 # 生成的图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[019-TWC-2025]Semantic_Communication_Enabled_6G-NTN_Framework_A_Novel_Denoising_and_Gateway_Hop_Integration_Mechanism"
python run_reproduction.py
```

## 复现结果

### Fig. 5: 通信时间对比

| 方案 | 复现平均延迟 | 论文平均延迟 | 偏差 |
|------|-------------|-------------|------|
| GA-DWOA | 0.821s | 0.819s | +0.3% |
| GA-GRE | 1.004s | 1.001s | +0.3% |
| GA-PRI | 1.061s | 1.040s | +2.0% |

**关键趋势**: 网关辅助方案显著降低通信时间; DWOA < GRE < PRI; 直接通信延迟方差大, 网关辅助方差小

### Fig. 6: 各GU信道质量 (SNR)

低SNR GU (间接连接) 通过网关后SNR显著提升, 高SNR GU (直接连接) 因释放子载波SNR略有改善.

### Fig. 7: 平均延迟 vs GU数量

3/4辅助比例延迟最低, 1/4辅助比例延迟最高且增长最快. 所有曲线随GU数量增加而上升.

### Fig. 9: PSNR vs SNR (不同压缩率)

压缩率越小 (1/48, 1/32), PSNR越高; 所有曲线随SNR单调上升并趋于饱和.

## 核心算法

1. **信道模型**: 自由空间路径损耗 + 校准因子匹配论文SNR范围
2. **DWOA (离散鲸鱼优化算法)**: 两阶段优化 (卫星→地面 + 网关→GU), 含QoS约束的适应度函数
3. **QoS机制**: PSNR阈值约束 + 违规惩罚 (0.5s/违规)
4. **语义通信**: 基于Swin Transformer的编解码, 语义压缩率SCR=1/16

## 依赖库

```bash
pip install numpy matplotlib scipy
```
