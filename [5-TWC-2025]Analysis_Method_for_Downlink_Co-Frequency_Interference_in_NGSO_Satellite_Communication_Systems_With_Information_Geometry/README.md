# TWC-2025 论文复现代码

## 论文信息

**标题**: Analysis Method for Downlink Co-Frequency Interference in NGSO Satellite Communication Systems With Information Geometry

**作者**: Yuanzhi He, Di Yan, Chengwu Qi

**发表**: IEEE Transactions on Wireless Communications, Vol. 24, No. 10, October 2025

## 项目结构

```
[5-TWC-2025]Analysis_Method_for.../
├── config.py              # 仿真参数配置 (Table I)
├── info_geometry.py       # 核心算法模块 (信息几何)
├── simulation.py          # 各图表仿真逻辑
├── plotting.py            # 绘图模块
├── run_reproduction.py    # 一键复现脚本
├── output/                # 输出图表
│   ├── fig3_airm_avg_dtc_vs_iterations.png
│   ├── fig4_airm_dtc_scatter_threshold.png
│   ├── fig5_skld_dtc_scatter_threshold.png
│   ├── fig6_jr_dtc_vs_sampling.png
│   ├── fig7_detection_probability.png
│   └── fig8_affine_embedding_3d.png
├── [5-TWC-2025]...pdf     # 论文原文
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[5-TWC-2025]Analysis_Method_for_Downlink_Co-Frequency_Interference_in_NGSO_Satellite_Communication_Systems_With_Information_Geometry"
python run_reproduction.py
```

## 复现结果

### Fig. 3: AIRM 平均 DTC 随迭代次数变化
- 展示 AIRM 度量下中心矩阵的迭代收敛过程
- 15 和 100 采样点下的平均 DTC 对比
- 迭代 20 次后基本收敛，与论文结论一致

### Fig. 4: AIRM 度量下的 DTC 散点图和阈值
- M=15 时 DTC 范围约 [0.10, 0.19]，阈值约 0.175
- M=100 时 DTC 分布更密集，阈值降低

### Fig. 5: SKLD 度量下的 DTC 散点图和阈值
- SKLD 度量下 DTC 值更小 (约 0.005-0.019)
- 随采样点增加分布更密集

### Fig. 6: JR-DTC 随采样点数变化
- 四种干扰场景 (Case0-3) 的 JR-DTC 区间
- AIRM 和 SKLD 两种度量下的对比

### Fig. 7: 正确判断概率 vs 采样点数
- AIRM 和 SKLD 两种度量在 4 种干扰场景下的检测概率
- 与能量检测法对比，信息几何方法更优
- 采样点超过 80 后检测概率趋于稳定 (>90%)

### Fig. 8: 仿射嵌入 3D 曲面图
- Starlink 卫星对 3 个 OneWeb 地球站的总干扰势函数
- 经纬度-势函数 3D 可视化

## 核心算法

论文将传统欧氏空间干扰分析转移到矩阵流形上：
1. 将接收信号建模为零均值高斯分布，其协方差矩阵构成矩阵流形
2. 在流形上寻找"中心矩阵"作为参考点（AIRM 迭代法 / SKLD 直接法）
3. 通过"到中心距离"(DTC) 判断是否存在干扰及干扰来源
4. 使用仿射嵌入将高维信号分布可视化到 3D 空间

## 依赖库

```bash
pip install numpy matplotlib scipy
```

