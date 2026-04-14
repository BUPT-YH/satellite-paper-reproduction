# -*- coding: utf-8 -*-
"""
一键复现脚本: 运行所有仿真并生成图表
论文: Cooperative Multi-Satellite and Multi-RIS Beamforming
期刊: IEEE JSAC 2025
"""

import os
import sys
import time

# 切换到项目目录
project_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_dir)
sys.path.insert(0, project_dir)

# 确保输出目录存在
os.makedirs('output', exist_ok=True)

from simulation import (
    fig2_simulation, fig3_simulation, fig4_simulation, fig5_simulation,
    fig6_simulation, fig7_simulation, fig8_simulation, fig9_simulation,
    fig10_simulation, fig11_simulation, fig12_simulation, fig13_simulation
)


def main():
    print("=" * 70)
    print("  论文复现: Cooperative Multi-Satellite and Multi-RIS Beamforming")
    print("  期刊: IEEE JSAC, Vol. 43, No. 1, 2025")
    print("=" * 70)

    total_start = time.time()
    figures = [
        ("Fig. 2: Min SINR vs PT (κN=20 dB)", fig2_simulation),
        ("Fig. 3: Min SINR vs PT (κN=0 dB)", fig3_simulation),
        ("Fig. 4: Min SINR vs ζ (κN=20 dB)", fig4_simulation),
        ("Fig. 5: Min SINR vs ζ (κN=0 dB)", fig5_simulation),
        ("Fig. 6: Min SINR vs M (κN=20 dB)", fig6_simulation),
        ("Fig. 7: Min SINR vs M (κN=0 dB)", fig7_simulation),
        ("Fig. 8: Min SINR vs κR (κN=0 dB)", fig8_simulation),
        ("Fig. 9: Min SINR vs κR (κN=10 dB)", fig9_simulation),
        ("Fig. 10: Min SINR vs κR (κN=20 dB)", fig10_simulation),
        ("Fig. 11: MSC vs SST Min SINR", fig11_simulation),
        ("Fig. 12: MSC vs SST Sum Rate", fig12_simulation),
        ("Fig. 13: Execution Time", fig13_simulation),
    ]

    completed = []
    failed = []

    for name, sim_func in figures:
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")
        try:
            t0 = time.time()
            result = sim_func()
            elapsed = time.time() - t0
            completed.append(name)
            print(f"  [OK] {name} completed in {elapsed:.1f}s")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"  [FAIL] {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  复现完成! 总用时: {total_time:.1f}s")
    print(f"  成功: {len(completed)}/{len(figures)}")
    if failed:
        print(f"  失败: {len(failed)}")
        for name, err in failed:
            print(f"    - {name}: {err[:100]}")
    print(f"  输出目录: {os.path.join(project_dir, 'output')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
