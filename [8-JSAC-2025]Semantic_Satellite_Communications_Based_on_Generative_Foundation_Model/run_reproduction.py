"""
一键复现脚本 — 生成论文中所有图表
运行: python run_reproduction.py
"""
import numpy as np
import os
import sys

# 确保在正确目录下运行
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from config import PROJECT
import simulation as sim
import plotting as plot
import semantic_methods as sm

OUTPUT_DIR = os.path.join(script_dir, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print(f"论文复现: {PROJECT}")
    print("Semantic Satellite Communications Based on Generative Foundation Model")
    print("IEEE JSAC, Vol.43, No.7, July 2025")
    print("=" * 60)
    print()

    # ============================================================
    # Fig. 7: Ploss 和 SSIM 性能对比
    # ============================================================
    print("[1/5] 生成 Fig. 7: Ploss 和 SSIM 性能对比...")
    ploss_data, ssim_data = sim.simulate_fig7()
    plot.plot_fig7(ploss_data, ssim_data, OUTPUT_DIR)

    # ============================================================
    # Fig. 9: 所需语义特征 Ploss
    # ============================================================
    print("[2/5] 生成 Fig. 9: 所需语义特征 Ploss 性能...")
    req_ploss_data = sim.simulate_fig9()
    plot.plot_fig9(req_ploss_data, OUTPUT_DIR)

    # ============================================================
    # Fig. 11: 错误检测器 MSE 性能
    # ============================================================
    print("[3/5] 生成 Fig. 11: 错误检测器 MSE 性能...")
    mse_data = sim.simulate_fig11()
    plot.plot_fig11(mse_data, OUTPUT_DIR)

    # ============================================================
    # Fig. 12: 错误检测器系统级性能 (柱状图)
    # ============================================================
    print("[4/5] 生成 Fig. 12: 错误检测器系统级性能 (柱状图)...")
    success_data, detection_data = sim.simulate_fig12()
    plot.plot_fig12(success_data, detection_data, OUTPUT_DIR)

    # ============================================================
    # Fig. 13: 消融实验
    # ============================================================
    print("[5/5] 生成 Fig. 13: 消融实验...")
    ablation_data = sim.simulate_fig13()
    plot.plot_fig13(ablation_data, OUTPUT_DIR)

    # ============================================================
    # 输出结果汇总
    # ============================================================
    print()
    print("=" * 60)
    print("复现完成! 所有图表已保存到 output/ 目录:")
    print("-" * 60)
    output_files = sorted(os.listdir(OUTPUT_DIR))
    for f in output_files:
        if f.endswith('.png'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size_kb:.1f} KB)")
    print("=" * 60)

    # 打印关键数值对比
    print()
    print("关键数值对比 (SNR=0 dB, No CCI):")
    print("-" * 40)
    snr_test = np.array([0.0])
    print(f"  JPEG+LDPC(64,127) Ploss: {float(sm.ploss_jpeg_ldpc(snr_test, 0)[0]):.4f}")
    print(f"  JSCC             Ploss: {float(sm.ploss_jscc(snr_test, 0)[0]):.4f}")
    print(f"  FMSAT(SegGPT)    Ploss: {float(sm.ploss_fmsat_seggpt(snr_test, 0)[0]):.4f}")
    print(f"  FMSAT(UNet)      Ploss: {float(sm.ploss_fmsat_unet(snr_test, 0)[0]):.4f}")
    print()
    print(f"  JPEG+LDPC(64,127) SSIM: {float(sm.ssim_jpeg_ldpc(snr_test, 0)[0]):.4f}")
    print(f"  JSCC              SSIM: {float(sm.ssim_jscc(snr_test, 0)[0]):.4f}")
    print(f"  FMSAT(SegGPT)     SSIM: {float(sm.ssim_fmsat_seggpt(snr_test, 0)[0]):.4f}")
    print("-" * 40)

    print()
    print("关键数值对比 (SNR=-5 dB, 0.5 CCI):")
    print("-" * 40)
    snr_bad = np.array([-5.0])
    print(f"  JSCC(Adapt)     Req Ploss: {float(sm.ploss_required_jscc_adapt(snr_bad, 0.5)[0]):.4f}")
    print(f"  AFMSAT          Req Ploss: {float(sm.ploss_required_afmsat(snr_bad, 0.5)[0]):.4f}")
    print(f"  AFMSAT(Correl)  Req Ploss: {float(sm.ploss_required_afmsat_correl(snr_bad, 0.5)[0]):.4f}")
    print("-" * 40)


if __name__ == '__main__':
    main()
