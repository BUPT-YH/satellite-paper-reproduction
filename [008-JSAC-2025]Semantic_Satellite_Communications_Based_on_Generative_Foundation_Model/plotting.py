"""
IEEE 期刊风格绘图模块
统一所有图表的字体、颜色、尺寸、标记样式
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os

# ===== IEEE 期刊风格全局配置 =====
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['lines.markersize'] = 5
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.minor.width'] = 0.5
rcParams['ytick.minor.width'] = 0.5
rcParams['grid.linewidth'] = 0.3
rcParams['grid.alpha'] = 0.3

# IEEE 图尺寸
FIG_SINGLE = (3.5, 2.8)    # 单栏图
FIG_DOUBLE = (7.16, 3.0)   # 双栏图
FIG_DOUBLE_TALL = (7.16, 3.5)

# 颜色方案 (高对比度, 灰度可区分)
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h']
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


def save_fig(fig, filename, output_dir='output'):
    """保存图表"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  已保存: {filepath}")


def plot_fig7(ploss_data, ssim_data, output_dir='output'):
    """
    Fig. 7: 不同信道条件下各方法的性能对比
    (a) Ploss performance  (b) SSIM performance
    """
    snr = ploss_data['snr']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # ---- (a) Ploss ----
    for method in ['JPEG+LDPC(64,127)', 'JPEG+LDPC(64,255)', 'JSCC', 'FMSAT(SegGPT)', 'FMSAT(UNet)']:
        for cci_label, cci_val in [('No_CCI', 0), ('0.5_CCI', 0.5)]:
            key = f"{method}_{cci_label}"
            if key in ploss_data:
                ax1.plot(snr, ploss_data[key],
                         color=ploss_data[key+'_color'],
                         marker=ploss_data[key+'_marker'],
                         linestyle=ploss_data[key+'_ls'],
                         label=ploss_data[key+'_label'],
                         markevery=3)

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Perceptual Loss (Ploss)')
    ax1.set_title('(a) Ploss Performance')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-10, 10])

    # ---- (b) SSIM ----
    for method in ['JPEG+LDPC(64,127)', 'JPEG+LDPC(64,255)', 'JSCC', 'FMSAT(SegGPT)', 'FMSAT(UNet)']:
        for cci_label, cci_val in [('No_CCI', 0), ('0.5_CCI', 0.5)]:
            key = f"{method}_{cci_label}"
            if key in ssim_data:
                ax2.plot(snr, ssim_data[key],
                         color=ssim_data[key+'_color'],
                         marker=ssim_data[key+'_marker'],
                         linestyle=ssim_data[key+'_ls'],
                         label=ssim_data[key+'_label'],
                         markevery=3)

    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('SSIM')
    ax2.set_title('(b) SSIM Performance')
    ax2.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-10, 10])
    ax2.set_ylim([0, 1.05])

    fig.tight_layout()
    save_fig(fig, 'fig7_ploss_ssim_performance.png', output_dir)


def plot_fig9(required_ploss_data, output_dir='output'):
    """
    Fig. 9: 所需语义特征的 Ploss 性能
    (a) No CCI  (b) 0.5 CCI
    """
    snr = required_ploss_data['snr']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    methods = ['FMSAT', 'AFMSAT', 'AFMSAT(Correl)', 'JSCC(Adapt)']
    colors = ['#7E2F8E', '#4DBEEE', '#A2142F', '#EDB120']
    markers = ['D', 'p', 'h', '^']
    linestyles = ['-', '-', '--', ':']

    # (a) No CCI
    for i, method in enumerate(methods):
        key = f"{method}_No_CCI"
        if key in required_ploss_data:
            ax1.plot(snr, required_ploss_data[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i],
                     label=method, markevery=3)

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Required Ploss')
    ax1.set_title('(a) No CCI')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-10, 10])

    # (b) 0.5 CCI
    for i, method in enumerate(methods):
        key = f"{method}_0.5_CCI"
        if key in required_ploss_data:
            ax2.plot(snr, required_ploss_data[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i],
                     label=method, markevery=3)

    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Required Ploss')
    ax2.set_title('(b) 0.5 CCI')
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-10, 10])

    fig.tight_layout()
    save_fig(fig, 'fig9_required_ploss_performance.png', output_dir)


def plot_fig11(mse_data, output_dir='output'):
    """
    Fig. 11: 错误检测器相关的 MSE 性能
    (a) 卫星端接收图像的 MSE  (b) 网关端接收图像的 MSE
    """
    snr = mse_data['snr']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    methods = ['AFMSAT', 'JSCC', 'JPEG+LDPC']
    colors = ['#4DBEEE', '#EDB120', '#0072BD']
    markers = ['p', '^', 'o']
    linestyles = ['-', '-.', '--']

    # (a) Satellite
    for i, method in enumerate(methods):
        key = f"{method}_satellite"
        if key in mse_data:
            ax1.plot(snr, mse_data[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i],
                     label=method, markevery=3)

    ax1.axhline(y=0.015, color='red', linestyle=':', linewidth=0.8, label='Threshold=0.015')
    ax1.set_xlabel('UL SNR (dB)')
    ax1.set_ylabel('MSE (Required Part)')
    ax1.set_title('(a) At Satellite')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-10, 10])

    # (b) Gateway
    for i, method in enumerate(methods):
        key = f"{method}_gateway"
        if key in mse_data:
            ax2.plot(snr, mse_data[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i],
                     label=method, markevery=3)

    ax2.set_xlabel('UL SNR (dB)')
    ax2.set_ylabel('MSE (Required Part)')
    ax2.set_title('(b) At Gateway (DL SNR=5 dB)')
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-10, 10])

    fig.tight_layout()
    save_fig(fig, 'fig11_mse_error_detector.png', output_dir)


def plot_fig12(success_data, detection_data, output_dir='output'):
    """
    Fig. 12: 错误检测器系统性能 — 柱状图
    (a) 成功传输率  (b) 粗检测器检出比例
    x 轴为离散信道条件组合
    """
    labels = success_data['labels']
    methods = success_data['methods']
    n_conditions = len(labels)
    n_methods = len(methods)

    colors = ['#4DBEEE', '#7E2F8E', '#EDB120', '#0072BD']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    # 柱状图参数
    total_width = 0.8
    bar_width = total_width / n_methods
    x = np.arange(n_conditions)

    # ---- (a) Success Rate ----
    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        values = [v * 100 for v in success_data[method]]
        ax1.bar(x + offset, values, bar_width, label=method,
                color=colors[i], edgecolor='black', linewidth=0.3, alpha=0.85)

    ax1.set_xlabel('Channel Condition (UL,DL SNR)')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('(a) Success Rate at Gateway')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.legend(loc='lower right', framealpha=0.9, edgecolor='gray', fontsize=7)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])

    # ---- (b) Detection Ratio ----
    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        values = [v * 100 for v in detection_data[method]]
        ax2.bar(x + offset, values, bar_width, label=method,
                color=colors[i], edgecolor='black', linewidth=0.3, alpha=0.85)

    ax2.set_xlabel('Channel Condition (UL,DL SNR)')
    ax2.set_ylabel('Detection Ratio (%)')
    ax2.set_title('(b) Rough Detector Ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=7)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    save_fig(fig, 'fig12_error_detection_performance.png', output_dir)


def plot_fig13(ablation_data, output_dir='output'):
    """
    Fig. 13: 消融实验
    (a) Ploss 性能  (b) Required Ploss 性能
    """
    snr = ablation_data['snr']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    configs = ['full', 'no_diffusion', 'no_segmentation', 'encoder_decoder_only']
    labels = ['Full (FMSAT)', 'w/o Diffusion', 'w/o Segmentation', 'Encoder-Decoder Only']
    colors = ['#7E2F8E', '#4DBEEE', '#77AC30', '#EDB120']
    markers = ['D', 'p', 'v', '^']
    linestyles = ['-', '--', '-.', ':']

    # (a) Ploss
    for i, cfg in enumerate(configs):
        key = f"{cfg}_ploss"
        if key in ablation_data:
            ax1.plot(snr, ablation_data[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i],
                     label=labels[i], markevery=3)

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Ploss')
    ax1.set_title('(a) Ploss Performance')
    ax1.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-10, 10])

    # (b) Required Ploss
    for i, cfg in enumerate(configs):
        key = f"{cfg}_required_ploss"
        if key in ablation_data:
            ax2.plot(snr, ablation_data[key],
                     color=colors[i], marker=markers[i], linestyle=linestyles[i],
                     label=labels[i], markevery=3)

    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Required Ploss')
    ax2.set_title('(b) Required Ploss Performance')
    ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-10, 10])

    fig.tight_layout()
    save_fig(fig, 'fig13_ablation_study.png', output_dir)
