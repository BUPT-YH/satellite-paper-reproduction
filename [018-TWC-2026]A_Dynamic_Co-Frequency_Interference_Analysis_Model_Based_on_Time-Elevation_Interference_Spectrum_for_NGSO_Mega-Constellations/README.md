# TWC-2026 论文复现代码

## 论文信息

**标题**: A Dynamic Co-Frequency Interference Analysis Model Based on Time-Elevation Interference Spectrum for NGSO Mega-Constellations

**作者**: Zhaoyang Su, Kai Wang, Yuchen Cai, Lipeng Ning, Liu Liu, Tao Zhou, Bo Ai

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[018-TWC-2026]A_Dynamic_Co-Frequency_Interference_Analysis_Model_Based_on_Time-Elevation_Interference_Spectrum_for_NGSO_Mega-Constellations/
├── [018-TWC-2026]...pdf    # 论文原文
├── config.py                # 仿真参数配置
├── constellation.py         # Walker星座建模、卫星位置计算、两阶段筛选
├── interference.py          # 路径损耗、天线增益、TEIS和INR计算
├── statistical.py           # INR概率密度、中断概率
├── plotting.py              # IEEE风格绘图模块
├── run_reproduction.py      # 一键复现脚本
├── output/                  # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[018-TWC-2026]A_Dynamic_Co-Frequency_Interference_Analysis_Model_Based_on_Time-Elevation_Interference_Spectrum_for_NGSO_Mega-Constellations"
python run_reproduction.py
```

## 复现结果

### Fig.5: Time-Elevation Interference Spectrum (TEIS)

- **(a) 倾角80°**: 清晰的周期性V形干扰热点，INR范围 [-16.1, 25.7] dB
- **(b) 倾角50°**: 干扰在仰角域更分散，热点重叠更紧密，INR范围 [-12.0, 25.7] dB

与论文对比: 周期性干扰模式和热点分布特征与原文Fig.5一致。

### Fig.9: INR概率密度函数

- 三种通信卫星仰角场景 (25°南/天顶/25°北) 的INR PDF
- 南向干扰略高于北向 (与论文分析一致)
- 天顶方向干扰最强，INR范围约 [-15, 10] dB

### Fig.11: 中断概率

- 基于TEIS计算中断概率，高中断概率区域与干扰热点位置对应
- 总采样点36600，高中断概率点6.6%

## 核心算法

1. **Walker星座参数化建模**: 通过参考卫星确定所有卫星位置 (Eq. 15-24)
2. **两阶段筛选**: 粗筛选(经纬度矩形域) + 精筛选(可见圆域) 快速识别干扰卫星 (Eq. 30-37)
3. **TEIS计算**: 联合时间-仰角域的聚合干扰表达 (Eq. 38-41)
4. **INR统计**: Monte Carlo方法验证INR概率分布 (Eq. 42-50)
5. **中断概率**: 基于SINR阈值的系统中断分析 (Eq. 13)

## 依赖库

```bash
pip install numpy matplotlib pymupdf pillow
```

## 简化假设

- 干扰星座为标准Walker Delta构型 (FI=1)
- 通信卫星采用简化模型 (单星从南向北过境)
- 每面卫星数22颗 (Starlink-like, 总计1584颗)
- 干扰终端分布半径500km, 最大100个终端
- 降雨衰减和大气吸收采用典型值简化计算
