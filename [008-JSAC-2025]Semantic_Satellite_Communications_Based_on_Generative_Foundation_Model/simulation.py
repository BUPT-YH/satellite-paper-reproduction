"""
主仿真逻辑 — 生成所有图表数据
"""
import numpy as np
from config import SNR_FINE, RANDOM_SEED
import semantic_methods as sm


def simulate_fig7():
    """
    Fig. 7: 各方法在不同信道条件下的 Ploss 和 SSIM 性能
    """
    np.random.seed(RANDOM_SEED)
    snr = SNR_FINE

    ploss_data = {'snr': snr}
    ssim_data = {'snr': snr}

    # 方法列表及样式
    methods_config = [
        ('JPEG+LDPC(64,127)', sm.ploss_jpeg_ldpc, sm.ssim_jpeg_ldpc,
         '#0072BD', 'o', '-', {'code_rate': 64/127}),
        ('JPEG+LDPC(64,255)', sm.ploss_jpeg_ldpc, sm.ssim_jpeg_ldpc,
         '#D95319', 's', '--', {'code_rate': 64/255}),
        ('JSCC', sm.ploss_jscc, sm.ssim_jscc,
         '#EDB120', '^', '-.', {}),
        ('FMSAT(SegGPT)', sm.ploss_fmsat_seggpt, sm.ssim_fmsat_seggpt,
         '#7E2F8E', 'D', '-', {}),
        ('FMSAT(UNet)', sm.ploss_fmsat_unet, sm.ssim_fmsat_unet,
         '#77AC30', 'v', '--', {}),
    ]

    for method, ploss_fn, ssim_fn, color, marker, ls, kwargs in methods_config:
        for cci_label, cci_val in [('No_CCI', 0), ('0.5_CCI', 0.5)]:
            key = f"{method}_{cci_label}"

            # Ploss
            ploss = ploss_fn(snr, cci_val, **kwargs)
            ploss += 0.005 * np.random.randn(len(snr)) * np.abs(ploss) * 0.05
            ploss = np.clip(ploss, 0, None)
            ploss_data[key] = ploss
            ploss_data[key + '_color'] = color
            ploss_data[key + '_marker'] = marker
            ploss_data[key + '_ls'] = ls if cci_val == 0 else '--'
            ploss_data[key + '_label'] = f"{method}" + ("" if cci_val == 0 else " (0.5 CCI)")

            # SSIM
            ssim = ssim_fn(snr, cci_val, **kwargs)
            ssim += 0.003 * np.random.randn(len(snr)) * 0.05
            ssim = np.clip(ssim, 0, 1)
            ssim_data[key] = ssim
            ssim_data[key + '_color'] = color
            ssim_data[key + '_marker'] = marker
            ssim_data[key + '_ls'] = ls if cci_val == 0 else '--'
            ssim_data[key + '_label'] = f"{method}" + ("" if cci_val == 0 else " (0.5 CCI)")

    return ploss_data, ssim_data


def simulate_fig9():
    """
    Fig. 9: 所需语义特征的 Ploss (仅计算重要区域)
    """
    np.random.seed(RANDOM_SEED + 1)
    snr = SNR_FINE

    data = {'snr': snr}

    methods = [
        ('FMSAT', sm.ploss_required_fmsat),
        ('AFMSAT', sm.ploss_required_afmsat),
        ('AFMSAT(Correl)', sm.ploss_required_afmsat_correl),
        ('JSCC(Adapt)', sm.ploss_required_jscc_adapt),
    ]

    for method, fn in methods:
        for cci_label, cci_val in [('No_CCI', 0), ('0.5_CCI', 0.5)]:
            key = f"{method}_{cci_label}"
            ploss = fn(snr, cci_val)
            ploss += 0.003 * np.random.randn(len(snr)) * np.abs(ploss) * 0.05
            ploss = np.clip(ploss, 0, None)
            data[key] = ploss

    return data


def simulate_fig11():
    """
    Fig. 11: 错误检测器相关的 MSE 性能
    """
    np.random.seed(RANDOM_SEED + 2)
    snr = SNR_FINE

    data = {'snr': snr}
    dl_snr = 5.0  # 下行 SNR 固定为 5 dB

    methods = ['AFMSAT', 'JSCC', 'JPEG+LDPC']

    for method in methods:
        # (a) 卫星端 MSE
        mse_sat = sm.mse_at_satellite(snr, 0.5, method)
        mse_sat += 0.001 * np.random.randn(len(snr)) * 0.05
        mse_sat = np.clip(mse_sat, 0, None)
        data[f"{method}_satellite"] = mse_sat

        # (b) 网关端 MSE
        mse_gw = sm.mse_at_gateway(snr, dl_snr, 0.5, method)
        mse_gw += 0.001 * np.random.randn(len(snr)) * 0.05
        mse_gw = np.clip(mse_gw, 0, None)
        data[f"{method}_gateway"] = mse_gw

    return data


def simulate_fig12():
    """
    Fig. 12: 错误检测器系统级性能 — 柱状图
    x 轴为离散的信道条件组合 (UL SNR, DL SNR, CCI)
    (a) 网关成功传输率
    (b) 粗检测器检出比例 (被网关拒绝的图像中, 粗检测器检出的比例)
    """
    np.random.seed(RANDOM_SEED + 3)

    # 离散信道条件: (UL SNR dB, DL SNR dB, CCI比例)
    channel_conditions = [
        (10, 10, 0),
        (5, 5, 0),
        (0, 5, 0),
        (-5, 5, 0),
        (10, 10, 0.5),
        (5, 5, 0.5),
        (0, 5, 0.5),
        (-5, 5, 0.5),
    ]
    condition_labels = [
        '10,10dB\nNo CCI',
        '5,5dB\nNo CCI',
        '0,5dB\nNo CCI',
        '-5,5dB\nNo CCI',
        '10,10dB\n0.5 CCI',
        '5,5dB\n0.5 CCI',
        '0,5dB\n0.5 CCI',
        '-5,5dB\n0.5 CCI',
    ]

    methods = ['AFMSAT', 'FMSAT', 'JSCC', 'JPEG+LDPC']

    success_data = {
        'conditions': channel_conditions,
        'labels': condition_labels,
        'methods': methods,
    }
    detection_data = {
        'conditions': channel_conditions,
        'labels': condition_labels,
        'methods': methods,
    }

    for method in methods:
        success_rates = []
        detect_ratios = []
        for ul_snr, dl_snr, cci in channel_conditions:
            # 计算端到端等效 SNR (用于成功率估计)
            # 再生卫星: 上行+下行误差累积
            snr_val = np.array([float(ul_snr)])
            rate = float(sm.success_rate_gateway(snr_val, cci, method)[0])
            # 添加小幅随机扰动
            rate += 0.01 * np.random.randn()
            rate = np.clip(rate, 0, 1)
            success_rates.append(rate)

            ratio = float(sm.rough_detector_detection_ratio(snr_val, cci, method)[0])
            ratio += 0.02 * np.random.randn()
            ratio = np.clip(ratio, 0, 1)
            detect_ratios.append(ratio)

        success_data[method] = success_rates
        detection_data[method] = detect_ratios

    return success_data, detection_data


def simulate_fig13():
    """
    Fig. 13: 消融实验
    """
    np.random.seed(RANDOM_SEED + 4)
    snr = SNR_FINE

    data = {'snr': snr}

    configs = ['full', 'no_diffusion', 'no_segmentation', 'encoder_decoder_only']

    for cfg in configs:
        # (a) Ploss
        ploss = sm.ploss_ablation(snr, 0.5, cfg)
        ploss += 0.003 * np.random.randn(len(snr)) * np.abs(ploss) * 0.05
        ploss = np.clip(ploss, 0, None)
        data[f"{cfg}_ploss"] = ploss

        # (b) Required Ploss
        req_ploss = sm.ploss_required_ablation(snr, 0.5, cfg)
        req_ploss += 0.002 * np.random.randn(len(snr)) * np.abs(req_ploss) * 0.05
        req_ploss = np.clip(req_ploss, 0, None)
        data[f"{cfg}_required_ploss"] = req_ploss

    return data
