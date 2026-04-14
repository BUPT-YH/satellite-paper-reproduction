# JSAC-2024 论文复现代码

## 论文信息

**标题**: Modeling Interference From Millimeter Wave and Terahertz Bands Cross-Links in Low Earth Orbit Satellite Networks for 6G and Beyond

**作者**: Sergi Aliaga Torrens, Vitaly Petrov, Josep Miquel Jornet

**发表**: IEEE Journal on Selected Areas in Communications (JSAC), Vol. 42, No. 5, May 2024

## 项目结构

```
[4-JSAC-2024]/
├── config.py              # 仿真参数配置 (Table II)
├── interference_model.py  # 干扰模型核心算法
├── plotting.py            # 绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[4-JSAC-2024]"
python run_reproduction.py
```

## 复现结果

### Figure 5(a): 单轨道 SIR vs 卫星数
SIR随卫星数增加而降低，所有曲线最终收敛到理论极限1.9 dB（Eq.10）。
波束宽度越窄，SIR越高。

### Figure 5(b): 单轨道 SINR vs 卫星数
mmWave链路在N=72处出现SINR断崖式下降（干扰卫星首次进入LOS）。
THz链路因窄波束基本不受干扰影响。

### Figure 5(c): 单轨道信道容量 vs 卫星数
mmWave容量极限约0.54 Gbps（论文~600 Mbps），sub-THz极限约13.5 Gbps（论文~15 Gbps）。

### Figure 9(b): 偏移轨道 SINR vs 波束宽度
展示了mmWave与sub-THz在不同波束宽度下的SINR差异。
mmWave因功率高SINR更好，但容量受带宽限制。

### Figure 10: 完整双星座部署容量
10轨道面×500卫星的双星座场景。sub-THz在N≈350后出现同轨道干扰，
mmWave在N≈70即受干扰影响。

## 核心算法

1. **单轨道干扰** (Eq. 1-10): 闭合形式SIR/SINR表达式
2. **共面轨道干扰** (Eq. 12-30): 时变干扰+时间平均
3. **偏移轨道干扰** (Eq. 31-45): 3D GEC坐标系建模
4. **天线模型**: 锥形辐射方向图 G = 2/(1-cos(α/2))

## 仿真参数 (Table II)

| 参数 | mmWave | sub-THz |
|------|--------|---------|
| 载波频率 | 38 GHz | 130 GHz |
| 带宽 | 400 MHz | 10 GHz |
| 发射功率 | 60 dBm (1000 W) | 27 dBm (0.5 W) |
| 系统温度 | 100 K | 100 K |

## 依赖库

```bash
pip install numpy matplotlib scipy PyMuPDF Pillow
```
