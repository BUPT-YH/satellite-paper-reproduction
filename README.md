# 📡 Satellite Paper Reproduction

> 卫星通信领域论文复现代码合集
>
> 来自公众号 **「静水Tech漫步」**

本仓库整理了卫星通信相关论文的复现代码，每篇论文对应一个独立目录，包含完整的代码和说明文档。

## 已复现论文

| # | 论文 | 期刊 | 关键词 | 目录 |
|---|------|------|--------|------|
| 1 | Avoiding Self-Interference in Megaconstellations Through Cooperative Satellite Routing and Frequency Assignment | JSAC 2024 | 星座自干扰避免、协作路由、频率分配 | [`[1-JSAC-2024]Avoiding_Self-Interference...`](./[1-JSAC-2024]Avoiding_Self-Interference_in_Megaconstellations_Through_Cooperative_Satellite_Routing_and_Frequency_Assignment) |
| 2 | Joint Power Allocation and Beam Scheduling in Beam-Hopping Satellites: A Two-Stage Framework With a Probabilistic Perspective | TWC 2024 | 波束跳变、功率分配、两阶段框架 | [`[2-TWC-2024]Joint_Power_Allocation...`](./[2-TWC-2024]Joint_Power_Allocation_and_Beam_Scheduling_in_Beam-Hopping_Satellites_A_Two-Stage_Framework_With_a_Probabilistic_Perspective) |
| 3 | Beam Footprint Design, Scheduling, and Spectrum Assignment in Low Earth Orbit Mega-Constellations | TVT 2025 | 波束足迹设计、用户调度、频谱分配 | [`[3-TVT-2025]Beam_Footprint_Design...`](./[3-TVT-2025]Beam_Footprint_Design_Scheduling_and_Spectrum_Assignment_in_Low_Earth_Orbit_Mega-Constellations) |
| 4 | Modeling Interference From Millimeter Wave and Terahertz Bands Cross-Links in Low Earth Orbit Satellite Networks for 6G and Beyond | JSAC 2024 | mmWave/THz星间链路、干扰建模、6G | [`[4-JSAC-2024]Modeling_Interference...`](./[4-JSAC-2024]Modeling_Interference_From_Millimeter_Wave_and_Terahertz_Bands_Cross-Links_in_Low_Earth_Orbit_Satellite_Networks_for_6G_and_Beyond) |
| 5 | Analysis Method for Downlink Co-Frequency Interference in NGSO Satellite Communication Systems With Information Geometry | TWC 2025 | 信息几何、共频干扰分析、矩阵流形 | [`[5-TWC-2025]Analysis_Method...`](./[5-TWC-2025]Analysis_Method_for_Downlink_Co-Frequency_Interference_in_NGSO_Satellite_Communication_Systems_With_Information_Geometry) |
| 6 | Feasibility Analysis of In-Band Coexistence in Dense LEO Satellite Communication Systems | TWC 2025 | 同频共存、干扰分析、卫星选择策略 | [`[6-TWC-2025]Feasibility_Analysis...`](./[6-TWC-2025]Feasibility_Analysis_of_In-Band_Coexistence_in_Dense_LEO_Satellite_Communication_Systems) |

## 快速使用

```bash
# 克隆仓库
git clone https://github.com/BUPU-YH/satellite-paper-reproduction.git

# 进入某篇论文的目录，按 README 说明运行
cd "[1-JSAC-2024]Avoiding_Self-Interference_in_Megaconstellations_Through_Cooperative_Satellite_Routing_and_Frequency_Assignment"
python run_reproduction.py
```

## 依赖安装

各论文所需依赖略有不同，主要包括：

```bash
pip install numpy matplotlib scipy scikit-learn
```

部分论文可能还需要 `torch`、`seaborn` 等，详见各目录下的 README。

## 关于

**「静水Tech漫步」** 专注于通信与信号处理领域的技术分享，包括：

- 卫星通信前沿论文解读
- 关键算法代码复现
- 仿真实验与性能分析

如果觉得有收获，欢迎 **Star** ⭐ 本仓库，后续更新第一时间收到！

## 关注我们

- **公众号**：静水Tech漫步
- **GitHub**：https://github.com/BUPU-YH

---

> 感谢关注「静水Tech漫步」！如果觉得有收获，欢迎点赞、在看、转发三连！
