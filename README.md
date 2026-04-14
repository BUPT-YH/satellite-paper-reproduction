# 📡 Satellite Paper Reproduction

> 卫星通信领域论文复现代码合集
>
> 来自公众号 **「静水Tech漫步」**

本仓库整理了卫星通信相关论文的复现代码，每篇论文对应一个独立目录，包含完整的代码和说明文档。

## 已复现论文

| # | 论文 | 期刊 | 关键词 | 目录 |
|---|------|------|--------|------|
| 001 | Avoiding Self-Interference in Megaconstellations Through Cooperative Satellite Routing and Frequency Assignment | JSAC 2024 | 星座自干扰避免、协作路由、频率分配 | [`[001-JSAC-2024]Avoiding_Self-Interference...`](./[001-JSAC-2024]Avoiding_Self-Interference_in_Megaconstellations_Through_Cooperative_Satellite_Routing_and_Frequency_Assignment) |
| 002 | Joint Power Allocation and Beam Scheduling in Beam-Hopping Satellites: A Two-Stage Framework With a Probabilistic Perspective | TWC 2024 | 波束跳变、功率分配、两阶段框架 | [`[002-TWC-2024]Joint_Power_Allocation...`](./[002-TWC-2024]Joint_Power_Allocation_and_Beam_Scheduling_in_Beam-Hopping_Satellites_A_Two-Stage_Framework_With_a_Probabilistic_Perspective) |
| 003 | Beam Footprint Design, Scheduling, and Spectrum Assignment in Low Earth Orbit Mega-Constellations | TVT 2025 | 波束足迹设计、用户调度、频谱分配 | [`[003-TVT-2025]Beam_Footprint_Design...`](./[003-TVT-2025]Beam_Footprint_Design_Scheduling_and_Spectrum_Assignment_in_Low_Earth_Orbit_Mega-Constellations) |
| 004 | Modeling Interference From Millimeter Wave and Terahertz Bands Cross-Links in Low Earth Orbit Satellite Networks for 6G and Beyond | JSAC 2024 | mmWave/THz星间链路、干扰建模、6G | [`[004-JSAC-2024]Modeling_Interference...`](./[004-JSAC-2024]Modeling_Interference_From_Millimeter_Wave_and_Terahertz_Bands_Cross-Links_in_Low_Earth_Orbit_Satellite_Networks_for_6G_and_Beyond) |
| 005 | Analysis Method for Downlink Co-Frequency Interference in NGSO Satellite Communication Systems With Information Geometry | TWC 2025 | 信息几何、共频干扰分析、矩阵流形 | [`[005-TWC-2025]Analysis_Method...`](./[005-TWC-2025]Analysis_Method_for_Downlink_Co-Frequency_Interference_in_NGSO_Satellite_Communication_Systems_With_Information_Geometry) |
| 006 | Feasibility Analysis of In-Band Coexistence in Dense LEO Satellite Communication Systems | TWC 2025 | 同频共存、干扰分析、卫星选择策略 | [`[006-TWC-2025]Feasibility_Analysis...`](./[006-TWC-2025]Feasibility_Analysis_of_In-Band_Coexistence_in_Dense_LEO_Satellite_Communication_Systems) |
| 007 | Resource Allocation and Load Balancing for Beam Hopping Scheduling in Satellite-Terrestrial Communications: A Cooperative Satellite Approach | TWC 2025 | 多星协作、跳波束调度、DRL资源分配、ISL负载均衡 | [`[007-TWC-2025]Resource_Allocation...`](./[007-TWC-2025]Resource_Allocation_and_Load_Balancing_for_Beam_Hopping_Scheduling_in_Satellite-Terrestrial_Communications_A_Cooperative_Satellite_Approach) |
| 008 | Semantic Satellite Communications Based on Generative Foundation Model | JSAC 2025 | 语义通信、生成式基础模型、知识图谱 | [`[008-JSAC-2025]Semantic_Satellite...`](./[008-JSAC-2025]Semantic_Satellite_Communications_Based_on_Generative_Foundation_Model) |
| 009 | Cooperative Multi-Satellite and Multi-RIS Beamforming: Enhancing LEO SatCom and Mitigating LEO-GEO Intersystem Interference | JSAC 2025 | 多星协作、RIS波束赋形、LEO-GEO干扰抑制、RZF预编码 | ⏳ waiting |
| 010 | LLM-Aided Spectrum-Sharing LEO Satellite Communications | JSAC 2026 | LLM辅助频谱共享、中断概率、资源分配 | ⏳ waiting |
| 011 | Achieving Covert Communications in Ultra-Dense LEO Satellite Systems by Exploiting Interference and Directional Uncertainty | JSAC 2026 | 隐蔽通信、干扰利用、方向不确定性 | ⏳ waiting |

## 快速使用

```bash
# 克隆仓库
git clone https://github.com/BUPT-YH/satellite-paper-reproduction.git

# 进入某篇论文的目录，按 README 说明运行
cd "[001-JSAC-2024]Avoiding_Self-Interference_in_Megaconstellations_Through_Cooperative_Satellite_Routing_and_Frequency_Assignment"
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
- **GitHub**：https://github.com/BUPT-YH

---

> 感谢关注「静水Tech漫步」！如果觉得有收获，欢迎点赞、在看、转发三连！
