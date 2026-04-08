"""
TWC-2024 论文复现 - 一键运行脚本
复现目标: Figure 4, Figure 11, Table V
"""
import sys
import time
import numpy as np

print("=" * 60)
print("TWC-2024 论文复现")
print("Joint Power Allocation and Beam Scheduling in")
print("Beam-Hopping Satellites: A Two-Stage Framework")
print("=" * 60)

from simulation import (
    run_convergence_multiple_densities,
    run_performance_comparison,
    run_table_v
)
from plotting import plot_figure4, plot_figure11, print_table_v

# ============================================================
# Figure 4: 收敛性分析
# ============================================================
print("\n[1/3] 运行 Figure 4: 收敛性分析...")
t0 = time.time()
convergence_results = run_convergence_multiple_densities(K=26)
fig4_path = plot_figure4(convergence_results)
print(f"  完成 (耗时 {time.time()-t0:.1f}s)")

# ============================================================
# Figure 11: 性能对比
# ============================================================
print("\n[2/3] 运行 Figure 11: 性能对比 (50 instances × 3 densities)...")
t0 = time.time()
performance_results = run_performance_comparison(K=26)
fig11_path = plot_figure11(performance_results)
print(f"  完成 (耗时 {time.time()-t0:.1f}s)")

# ============================================================
# Table V: 定量对比
# ============================================================
print("\n[3/3] 运行 Table V: 定量性能对比...")
t0 = time.time()
table_data = run_table_v(K=26)
print_table_v(table_data)
print(f"  完成 (耗时 {time.time()-t0:.1f}s)")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("复现完成!")
print(f"Figure 4: {fig4_path}")
print(f"Figure 11: {fig11_path}")
print("Table V:  已打印到控制台")
print("=" * 60)
