"""
一键复现脚本
"""

import sys, os, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from config import STARLINK_SHELLS, KUIPER_SHELLS
from constellation import build_constellation
from simulation import run_simulation
import plotting
import antenna


def main():
    print("=" * 60)
    print("论文复现: Feasibility Analysis of In-Band Coexistence")
    print("         in Dense LEO Satellite Communication Systems")
    print("=" * 60)
    t0 = time.time()

    print("\n[1] 构建星座...")
    sl = build_constellation(STARLINK_SHELLS)
    kp = build_constellation(KUIPER_SHELLS)
    print(f"  Starlink: {len(sl[0])}, Kuiper: {len(kp[0])}")

    print("\n[2] Fig. 2-3 (波束图 + SE 损失)...")
    plotting.plot_fig2(antenna)
    plotting.plot_fig3()

    print("\n[3] 主仿真...")
    R = run_simulation(sl, kp, '32x32', verbose=True)

    print("\n[4] 生成仿真图表...")
    plotting.plot_fig4(R)
    plotting.plot_fig5(R)
    plotting.plot_fig6(R, '32x32')
    plotting.plot_fig7(R)
    plotting.plot_fig8(R)
    plotting.plot_fig9(R)
    plotting.plot_fig10(R)
    plotting.plot_fig11(R)
    plotting.plot_fig13(R)
    plotting.plot_fig14(R)

    print(f"\n{'='*60}")
    print(f"完成! 耗时 {((time.time()-t0)/60):.1f} 分钟")
    print(f"图表: {os.path.abspath('output')}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
