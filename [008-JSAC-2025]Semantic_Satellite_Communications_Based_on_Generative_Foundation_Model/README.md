# JSAC-2025 论文复现代码

## 论文信息

**标题**: Semantic Satellite Communications Based on Generative Foundation Model

**作者**: Peiwen Jiang, Chao-Kai Wen, Xiao Li, Shi Jin, Geoffrey Ye Li

**发表**: IEEE Journal on Selected Areas in Communications (JSAC), Vol. 43, No. 7, July 2025

## 项目结构

```
[8-JSAC-2025]Semantic_Satellite_Communications_Based_on_Generative_Foundation_Model/
├── config.py              # 仿真参数配置
├── channel_model.py       # LEO卫星信道模型 (路径损耗/多径/多普勒/CCI)
├── semantic_methods.py    # 语义通信方法性能模型
├── simulation.py          # 主仿真逻辑
├── plotting.py            # IEEE风格绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[8-JSAC-2025]Semantic_Satellite_Communications_Based_on_Generative_Foundation_Model"
python run_reproduction.py
```

## 复现结果

### Fig. 7: Ploss 和 SSIM 性能对比
不同信道条件下各方法的性能:
- JPEG+LDPC 在高SNR时性能最优，但存在明显的悬崖效应
- FMSAT(SegGPT) 的 Ploss 持续优于 JSCC
- 在恶劣条件下 (CCI + 低SNR)，FMSAT 的 SSIM 超过 JSCC

### Fig. 9: 所需语义特征 Ploss
- AFMSAT 在低SNR时优于 FMSAT (聚焦重要部分)
- AFMSAT(Correl) 利用先前图像相关性，性能最优
- 高SNR时 AFMSAT 略逊于 FMSAT (因自适应编码器针对0dB优化)

### Fig. 11: 错误检测器 MSE 性能
- 卫星端 MSE 随上行SNR提升而降低
- 网关端 MSE 较低 (下行链路条件较好)

### Fig. 12: 错误检测器系统性能 (柱状图)
- x 轴为离散信道条件组合 (UL/DL SNR + CCI)
- AFMSAT 成功率在各种条件下保持最高
- 粗检测器在低SNR时检测约50%的错误，节省带宽

### Fig. 13: 消融实验
- 完整框架 (分割+编解码+扩散重建) 性能最优
- 扩散模型贡献最大 (去掉后 Ploss 显著增加)
- 语义分割对重要区域保护有显著贡献

## 核心算法

论文提出 FMSAT 框架，核心包含:
1. **SegGPT/UNet 语义分割**: 提取图像重要语义特征
2. **自适应多速率编码器-解码器**: 根据信道条件选择最优编解码对
3. **条件扩散模型重建**: 从受损特征恢复高质量图像
4. **双层错误检测器**: 卫星粗检测 + 网关精细检测
5. **上下文相关性**: 利用先前图像修复当前受损图像

## 复现说明

由于原论文依赖训练好的深度学习模型 (SegGPT, UNet, 扩散模型, JSCC编解码器)，本复现采用参数化模型拟合论文趋势:
- 各方法的性能曲线基于论文描述的趋势进行数学建模
- 曲线形状、相对关系和关键转折点与论文一致
- 数值量级在合理范围内，可用于趋势分析和教学演示

## 依赖库

```bash
pip install numpy matplotlib scipy
```
