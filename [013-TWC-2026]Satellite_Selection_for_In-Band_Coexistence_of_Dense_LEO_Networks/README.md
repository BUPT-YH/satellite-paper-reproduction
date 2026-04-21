# TWC-2026 论文复现代码

## 论文信息

**标题**: Satellite Selection for In-Band Coexistence of Dense LEO Networks

**作者**: Eunsun Kim, Ian P. Roberts, Taekyun Lee, Jeffrey G. Andrews

**发表**: IEEE Transactions on Wireless Communications, Vol. 25, 2026

## 项目结构

```
[013-TWC-2026]Satellite_Selection_for_In-Band_Coexistence_of_Dense_LEO_Networks/
├── config.py              # 精确 FCC 轨道参数 (Starlink 6壳层 + Kuiper 3壳层)
├── satellite_selection.py # 星座建模与信道计算 (3GPP 天线模型)
├── simulation.py          # 主仿真逻辑 (基线/优化/敏感性)
├── plotting.py            # IEEE 风格绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[013-TWC-2026]Satellite_Selection_for_In-Band_Coexistence_of_Dense_LEO_Networks"
python run_reproduction.py
```

## 精确 FCC 轨道参数

### Starlink (主系统, 6900 颗, 6 壳层)

| 壳层 | 高度 (km) | 倾角 (°) | 轨道面 | 每面卫星 | 合计 |
|------|----------|---------|--------|---------|------|
| 1 | 540 | 53.2 | 72 | 22 | 1,584 |
| 2 | 550 | 53.0 | 72 | 22 | 1,584 |
| 3 | 560 | 97.6 | 4 | 43 | 172 |
| 4 | 560 | 97.6 | 6 | 58 | 348 |
| 5 | 570 | 70.0 | 36 | 20 | 720 |
| 6 | 530 | 33.0 | 28 | 89 | 2,492 |

### Kuiper (次系统, 3236 颗, 3 壳层)

| 壳层 | 高度 (km) | 倾角 (°) | 轨道面 | 每面卫星 | 合计 |
|------|----------|---------|--------|---------|------|
| 1 | 630 | 51.9 | 34 | 34 | 1,156 |
| 2 | 610 | 51.9 | 36 | 36 | 1,296 |
| 3 | 590 | 33.0 | 28 | 28 | 784 |

## 仿真参数 (Table III)

| 参数 | 值 |
|------|-----|
| 卫星天线 | 64×64, 1.6° beamwidth, 36 dBi |
| 用户天线 | 32×32, 3.2° beamwidth, 30 dBi |
| 主系统 EIRP | -54.3 dBW/Hz |
| 次系统 EIRP | -53.3 dBW/Hz |
| 噪声 PSD | -174 dBm/Hz |
| 噪声系数 | 1.2 dB |
| 载频 | 12 GHz (Ka) |
| 带宽 | 250 MHz |
| 小区半径 | 10 km |

## 复现结果

### Fig. 4: 基线 INR CDF
- NB=8 median -9.8 dB, NB=32 median -3.7 dB (每翻倍 +3 dB)
- HE/MCT 策略差异 ~2.5 dB
- 趋势与论文一致

### Fig. 5: 提出方案 INR CDF
- INRmax=-6: median -7.0 dB (严格约束降低干扰)
- INRmax=∞: median -6.8 dB

## 依赖库

```bash
pip install numpy matplotlib
```
