"""
一键复现脚本
复现论文 Fig.5 (TEIS), Fig.9 (INR PDF), Fig.11 (中断概率)
"""
import os
import sys
import numpy as np

# 切换到项目目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from config import *
from constellation import (
    get_visible_satellites, get_interfering_terminals,
    walker_satellite_positions,
)
from interference import compute_teis
from statistical import (
    compute_inr_pdf_monte_carlo, compute_outage_probability
)
from plotting import plot_teis, plot_inr_pdf, plot_outage_heatmap, COLORS

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def reproduce_fig5():
    """
    复现 Fig.5: TEIS for INR
    (a) inclination = 80°
    (b) inclination = 50°
    """
    print('\n' + '='*60)
    print('复现 Fig.5: Time-Elevation Interference Spectrum (TEIS)')
    print('='*60)

    # 时间和仰角范围
    t_array = np.arange(0, T_SIM, DT)
    elev_array = np.arange(0, 181, 3)  # 0°(南地平线) → 90°(天顶) → 180°(北地平线)

    # 生成固定干扰终端分布
    int_terminals, _, _ = get_interfering_terminals(
        GS_LAT, GS_LON, INT_RADIUS, MAX_INT_TERMINALS, seed=42
    )
    print(f'  干扰终端数: {len(int_terminals)}')

    for incl, label in [(INCLINATION_1, '80deg'), (INCLINATION_2, '50deg')]:
        print(f'\n  计算 TEIS (inclination={incl}°)...')
        teis = compute_teis(
            GS_LAT, GS_LON, H_ORBIT, incl,
            N_PLANES, N_SATS_PER_PLANE,
            t_array, elev_array, int_terminals, THETA_MIN
        )
        np.save(f'{OUTPUT_DIR}/teis_incl_{label}.npy', teis)

        fname = f'{OUTPUT_DIR}/fig5_{label}_teis.png'
        plot_teis(t_array, elev_array, teis,
                  f'TEIS for INR (Inclination={incl}°)', fname)
        print(f'  INR范围: [{teis.max():.1f}, {teis.min():.1f}] dB')


def reproduce_fig9():
    """
    复现 Fig.9: INR PDF at different elevation angles
    对比三种通信卫星位置: 25°南, 天顶(90°), 25°北
    """
    print('\n' + '='*60)
    print('复现 Fig.9: INR PDF at different elevation angles')
    print('='*60)

    n_mc = 2000  # Monte Carlo采样数
    incl = INCLINATION_1  # 80°

    # 论文中的三种仰角场景
    # elevation in figure: 0=南地平线, 90=天顶, 180=北地平线
    # 25°南 → elev=25, 天顶 → elev=90, 25°北 → elev=155
    scenarios = [
        (25.0, '25° South'),
        (90.0, 'Zenith (90°)'),
        (155.0, '25° North'),
    ]

    inr_samples_list = []
    labels = []
    colors_list = [COLORS[0], COLORS[1], COLORS[2]]

    for elev, label in scenarios:
        print(f'\n  Monte Carlo仿真 (elevation={label}, N={n_mc})...')
        samples = compute_inr_pdf_monte_carlo(
            GS_LAT, GS_LON, H_ORBIT, incl,
            N_PLANES, N_SATS_PER_PLANE, elev,
            INT_RADIUS, MAX_INT_TERMINALS, n_mc=n_mc
        )
        inr_samples_list.append(samples)
        labels.append(label)
        print(f'  INR范围: [{samples[samples > -25].min():.1f}, {samples[samples > -25].max():.1f}] dB')

    fname = f'{OUTPUT_DIR}/fig9_inr_pdf.png'
    plot_inr_pdf(inr_samples_list, labels, colors_list, fname,
                 title='INR PDF at Different Elevation Angles')


def reproduce_fig11():
    """
    复现 Fig.11: Outage Probability
    基于Fig.5(a)的TEIS结果计算中断概率
    """
    print('\n' + '='*60)
    print('复现 Fig.11: Outage Probability')
    print('='*60)

    # 加载TEIS数据 (80°倾角)
    teis_file = f'{OUTPUT_DIR}/teis_incl_80deg.npy'
    if os.path.exists(teis_file):
        teis = np.load(teis_file)
    else:
        print('  需要先运行 Fig.5 复现, 生成TEIS数据...')
        return

    t_array = np.arange(0, T_SIM, DT)
    elev_array = np.arange(0, 181, 3)

    print('  计算中断概率...')
    outage = compute_outage_probability(
        teis, GS_LAT, GS_LON, H_ORBIT, elev_array
    )

    fname = f'{OUTPUT_DIR}/fig11_outage_probability.png'
    plot_outage_heatmap(t_array, elev_array, outage,
                        'Outage Probability (Inclination=80°)', fname)

    # 统计
    total = outage.size
    outage_count = np.sum(outage > 0.5)
    print(f'  总采样点: {total}, 高中断概率点: {outage_count} ({outage_count/total*100:.1f}%)')


def extract_scene_figures():
    """提取场景图/架构图 (Fig.1-3) 从PDF"""
    import fitz
    from PIL import Image

    pdf_path = None
    for f in os.listdir('.'):
        if f.endswith('.pdf'):
            pdf_path = f
            break

    if pdf_path is None:
        print('  未找到PDF文件')
        return

    print(f'  提取场景图: {pdf_path}')
    doc = fitz.open(pdf_path)

    # 论文中的场景图位置 (IEEE双栏格式, letter size 612×792 pt)
    # Fig.1 (page 3): 右栏 x=[312,563], caption y=253pt, 图在header(38pt)到caption之间
    # Fig.2 (page 5): 右栏 x=[312,563], caption y=266pt
    # Fig.3 (page 7): 左栏 x=[49,300], caption y=197pt
    figure_specs = {
        2: [  # Page 3 (0-based index) - 右栏
            ('fig1_interference_scenario', 312, 38, 563, 253),
        ],
        4: [  # Page 5 - 右栏
            ('fig2_constellation_config', 312, 38, 563, 266),
        ],
        6: [  # Page 7 - 左栏
            ('fig3_filtering_diagram', 49, 38, 300, 197),
        ],
    }

    scale = 300 / 72
    for page_num, specs in figure_specs.items():
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

        for name, x0, y0, x1, y1 in specs:
            left = int(x0 * scale)
            top = int(y0 * scale)
            right = int(x1 * scale)
            bottom = int(y1 * scale)
            # 边界检查
            left = max(0, min(left, img.width))
            top = max(0, min(top, img.height))
            right = max(0, min(right, img.width))
            bottom = max(0, min(bottom, img.height))

            cropped = img.crop((left, top, right, bottom))
            cropped.save(f'{OUTPUT_DIR}/{name}.png', dpi=(300, 300))
            print(f'  Saved: {OUTPUT_DIR}/{name}.png ({cropped.size[0]}x{cropped.size[1]})')

    doc.close()


if __name__ == '__main__':
    print('='*60)
    print('论文复现: TEIS-based Dynamic CFI Analysis for NGSO')
    print('IEEE TWC 2026')
    print('='*60)

    # Step 0: 提取场景图
    print('\n[Step 0] 提取场景图/架构图...')
    try:
        extract_scene_figures()
    except Exception as e:
        print(f'  场景图提取失败: {e}')

    # Step 1: 复现 Fig.5 (TEIS)
    reproduce_fig5()

    # Step 2: 复现 Fig.9 (INR PDF)
    reproduce_fig9()

    # Step 3: 复现 Fig.11 (中断概率)
    reproduce_fig11()

    print('\n' + '='*60)
    print('复现完成! 所有结果保存在 output/ 目录')
    print('='*60)
