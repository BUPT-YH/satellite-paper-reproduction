# TMC-2026 论文复现代码

## 论文信息

**标题**: Downlink Performance of Cell-Free Massive MIMO for LEO Satellite Mega-Constellation

**作者**: Xiangyu Li, Bodong Shang

**发表**: IEEE Transactions on Mobile Computing, Vol. 25, No. 4, 2026

## 项目结构

```
[027-TMC-2026]Downlink_Performance_of_Cell-Free_Massive_MIMO_for_LEO_Satellite_Mega-Constellation/
├── config.py              # 仿真参数配置 (Table I)
├── simulation.py          # Monte Carlo仿真引擎 + 解析表达式
├── plotting.py            # IEEE期刊风格绘图模块
├── run_reproduction.py    # 一键复现脚本 (全部10张数据图)
├── output/                # 输出图表 (13张)
│   ├── fig1_constellations.png                # 星座系统模型 (论文原图)
│   ├── fig2_stochastic_geometry.png           # 随机几何模型 (论文原图)
│   ├── fig3_distance_statistics.png           # 距离统计 (论文原图)
│   ├── fig4_coverage_ppp_vs_starlink.png      # PPP vs Starlink覆盖概率
│   ├── fig5_dss_ccdf.png                      # DSS的CCDF (完美/非完美CSI)
│   ├── fig6_coverage_nakagami_m.png           # 不同Nakagami m参数的覆盖概率
│   ├── fig7_cf_vs_cell.png                    # CF方案 vs Cell-based方案
│   ├── fig8_with_without_beamforming.png      # 有/无波束赋形对比
│   ├── fig9_coverage_dome_angle.png           # 不同圆顶角η的覆盖概率
│   ├── fig10_coverage_3d_altitude_saps.png    # 3D覆盖概率 (轨道高度 vs SAP数)
│   ├── fig11_coverage_3d_saps_uts.png         # 3D覆盖概率 (SAP数 vs UT数)
│   ├── fig12_system_capacity.png              # 系统容量 vs UT数
│   └── fig13_per_user_capacity.png            # 每用户容量 vs UT数
└── README.md
```

## 快速开始

```bash
cd "C:\Users\windows\Desktop\文章复现\[027-TMC-2026]Downlink_Performance_of_Cell-Free_Massive_MIMO_for_LEO_Satellite_Mega-Constellation"
python run_reproduction.py
```

一键运行即可生成全部13张图（含3张从PDF提取的概念图和10张仿真数据图）。完整运行约需30-60分钟。

## 复现结果

### Fig. 4: PPP模型 vs Starlink星座覆盖概率
- 随机几何PPP模型给出的覆盖概率是Starlink星座在不同纬度下的统计平均
- Starlink在低纬度（20°）和中纬度（40°）覆盖优于高纬度（60°）
- 随机初始相位（R）与固定相位（F）星座的覆盖行为差异

### Fig. 5: DSS的CCDF
- 完美CSI下DSS的CCDF呈现典型的右偏分布
- 非完美CSI（LMMSE信道估计，tau_p=20/100/200）下DSS衰减更快
- tau_p越大（导频越长），信道估计越准确，DSS性能越接近完美CSI

### Fig. 6: 不同Nakagami-m参数的覆盖概率
- m越大（信道条件越好），覆盖概率越高
- λU增大（用户更密集）导致每SAP服务更多UT，多用户干扰增加，覆盖下降
- 解析结果与Monte Carlo仿真吻合

### Fig. 7: Cell-Free vs Cell-based方案
- Cell-Free方案（多SAP协作服务UT）显著优于Cell-based（仅最近SAP服务）
- SAP密度越大，Cell-Free增益越明显
- Cell-based方案受限于单点服务，干扰无法被协作抑制

### Fig. 8: 有/无波束赋形对比
- 波束赋形（BF）显著提升覆盖概率，验证了相干合并增益
- m=4（信道条件好）时BF增益大于m=1
- 无BF时信号功率非相干叠加，无法获得阵列增益

### Fig. 9: 不同圆顶角η的覆盖概率
- 存在crossover现象：低SINR门限时η越大覆盖越好，高SINR门限时η越大覆盖反而变差
- η增大带来更多服务SAP参与协作（DS增益），但也引入更多多用户干扰（MUI）
- 不存在全局最优圆顶角，取决于目标SINR工作点

### Fig. 10: 3D覆盖概率（轨道高度 vs SAP数量）
- 固定SAP数量下，低轨道高度（400km）覆盖优于高轨道（1000km）
- SAP数量增加显著提升覆盖概率
- 轨道高度与SAP数量存在最优折中

### Fig. 11: 3D覆盖概率（SAP数量 vs UT数量）
- SAP数量增加提升覆盖，UT数量增加降低覆盖（干扰增加）
- 高SAP密度下，系统对UT密度变化更鲁棒

### Fig. 12: 系统容量 vs UT数量
- 存在最优UT数量使系统容量最大化，超过临界点后容量下降
- Cell-Free方案显著优于Nearest方案（最近卫星），尤其在大用户规模时
- 低轨道高度（500km）优于高轨道（1000km），大η（80°）在小UT规模下占优

### Fig. 13: 每用户容量 vs UT数量
- Cell-Free方案每用户容量随UT数缓慢下降（多用户干扰）
- Nearest方案每用户容量下降更剧烈
- η=80°配置下每用户容量最优

## 核心算法

### 系统模型
基于随机几何（Stochastic Geometry）的LEO卫星巨型星座下行性能分析框架：
- SAP在球面上按PPP分布，UT在地球表面按PPP分布
- 典型UT位于北极点（0, 0, RE），利用Slivnyak定理简化分析

### 关键公式

1. **距离边界**（Eq.10-11）: 服务SAP距离范围 [rS_min, rS_max]，由圆顶角η确定
   - rS_min = RS - RE
   - rS_max = sqrt(RS² - RE²sin²η) - RE·cosη

2. **每SAP平均UT数**（Eq.12）: |ΦU| = 2πRE·λU·(RE - RE²sin²η/RS - RE·sqrt(RS² - RE²sin²η)·cosη/RS)

3. **覆盖概率**（Theorem 1, Eq.31）: 通过Laplace变换和Gil-Pelaez反演求解
   - 将DSS（期望信号强度）的CDF/CCDF转换为Laplace域数值积分

4. **SINR分解**:
   - 期望信号（DS）: 多SAP相干合并，MRT波束赋形
   - 多用户干扰（MUI）: 同一SAP服务多个UT导致的干扰
   - 星间干扰（ISI）: 非服务区域SAP通过旁瓣产生的干扰

5. **系统容量**（Eq.33-35）: C = N_U·B·(τ_c - τ_p)/τ_c · E[log2(1+SINR)]

### 仿真方法
- Monte Carlo仿真：每次生成球面PPP分布的SAP，计算典型UT的SINR
- 解析计算：通过Gil-Pelaez反演公式将Laplace变换转为CDF
- LMMSE信道估计：模拟非完美CSI场景（tau_p导频长度影响估计精度）

## 仿真参数 (Table I)

| 参数 | 符号 | 值 |
|------|------|------|
| 地球半径 | RE | 6371.393 km |
| 轨道高度 | HS | 500 km |
| 载波频率 | fc | 2 GHz |
| 带宽 | B | 30 MHz |
| 下行发射功率 | ρd | 33 dBm |
| 导频发射功率 | ρp | 30 dBm |
| 噪声功率 | σ² | -100 dBm |
| SAP主瓣增益 | Gml | 30 dBi |
| SAP旁瓣增益 | Gsl | 20 dBi |
| UT接收增益 | Gr | 0 dBi |
| Nakagami衰落参数 | m | 2 |
| 路径损耗指数 | α | 2 |
| 圆顶角 | η | 75° |
| 导频长度 | τp | 200 |
| 相干块长度 | τc | 500 |
| SAP密度 | λS | 1×10⁻⁵ /km² |
| UT密度 | λU | 3×10⁻⁶ /km² |

## 依赖库

```bash
pip install numpy scipy matplotlib
```

可选（用于提取论文概念图）:
```bash
pip install pymupdf pillow
```

## 备注

- Monte Carlo仿真次数：覆盖概率10000次、容量2000次、3D曲面3000次
- 解析表达式中的Gil-Pelaez反演使用A=20, B=15, C=15参数确保数值精度
- Fig.4中Starlink星座使用简化Walker星座模型（3倾角层×28轨道面）
- 路径损耗指数α=2对应自由空间传播（卫星信道典型值）
- 有效天线增益G已包含频率因子 (c/4πfc)²，单位为km²
