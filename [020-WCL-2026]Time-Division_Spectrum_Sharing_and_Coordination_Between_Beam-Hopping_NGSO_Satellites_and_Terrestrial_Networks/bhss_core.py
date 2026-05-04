"""
BHSS 核心算法模块
实现论文 Algorithm 1: BHSS Time Alignment and Configuration
正确区分干扰时隙内部（完全不可用）和边界（特殊时隙部分可用）
"""
import numpy as np
from config import (
    ALTITUDE, NUM_VISIBLE_SATS, NUM_BEAMS_PER_SAT, NUM_TERR_CELLS,
    NUM_SAT_CELLS, W, GP_S2C, GP_C2S, T_SYM, N_SYM, CELL_RADIUS,
    AREA_SIZE, BEAM_WIDTH_3DB, L
)

C = 3e8


def compute_beam_footprint_radius():
    """波束足迹半径 (km)"""
    return ALTITUDE * np.tan(np.radians(BEAM_WIDTH_3DB / 2))


def generate_scenario():
    """生成仿真场景: 卫星、小区、BH模式、干扰关系"""
    np.random.seed(42)
    footprint_r = compute_beam_footprint_radius()
    half = AREA_SIZE / 2

    # 地面小区位置 (2D, km)
    terr_pos = np.random.uniform(-half, half, size=(NUM_TERR_CELLS, 2))

    # 卫星小区位置 (每颗卫星覆盖一个子区域)
    cells_per_sat = NUM_SAT_CELLS // NUM_VISIBLE_SATS
    sat_cell_pos = np.zeros((NUM_SAT_CELLS, 2))
    for s in range(NUM_VISIBLE_SATS):
        cx = np.random.uniform(-half * 0.7, half * 0.7)
        cy = np.random.uniform(-half * 0.7, half * 0.7)
        start = s * cells_per_sat
        end = start + cells_per_sat
        sat_cell_pos[start:end, 0] = cx + np.random.uniform(-40, 40, cells_per_sat)
        sat_cell_pos[start:end, 1] = cy + np.random.uniform(-40, 40, cells_per_sat)

    # 干扰关系 O[s][n][m]: 卫星 s 的波束指向卫星小区 n 时是否干扰地面小区 m
    O = np.zeros((NUM_VISIBLE_SATS, NUM_SAT_CELLS, NUM_TERR_CELLS), dtype=int)
    for s in range(NUM_VISIBLE_SATS):
        for n in range(NUM_SAT_CELLS):
            for m in range(NUM_TERR_CELLS):
                dist = np.sqrt((sat_cell_pos[n, 0] - terr_pos[m, 0])**2 +
                               (sat_cell_pos[n, 1] - terr_pos[m, 1])**2)
                if dist < footprint_r + CELL_RADIUS:
                    O[s, n, m] = 1

    # BH 模式
    np.random.seed(100)
    bh_patterns = {}
    for s in range(NUM_VISIBLE_SATS):
        own_cells = list(range(s * cells_per_sat, (s + 1) * cells_per_sat))
        pattern = np.zeros((W, NUM_BEAMS_PER_SAT), dtype=int)
        for t in range(W):
            chosen = np.random.choice(own_cells, size=NUM_BEAMS_PER_SAT, replace=False)
            pattern[t] = chosen
        bh_patterns[s] = pattern

    # 星地时延
    delays = np.zeros((NUM_VISIBLE_SATS, NUM_TERR_CELLS))
    for s in range(NUM_VISIBLE_SATS):
        # 卫星在区域上方, 简化位置
        for m in range(NUM_TERR_CELLS):
            dist_m = np.sqrt(terr_pos[m, 0]**2 + terr_pos[m, 1]**2 + ALTITUDE**2) * 1e3
            delays[s, m] = dist_m / C

    # 受干扰小区
    disturbed_terr = np.any(O == 1, axis=(0, 1))
    disturbed_terr_idx = np.where(disturbed_terr)[0]
    disturbed_sat = np.zeros(NUM_SAT_CELLS, dtype=bool)
    for s in range(NUM_VISIBLE_SATS):
        for n in range(NUM_SAT_CELLS):
            if np.any(O[s, n, :] == 1):
                disturbed_sat[n] = True
    disturbed_sat_idx = np.where(disturbed_sat)[0]

    # 每个地面小区每时隙的干扰卫星数
    interference_count = np.zeros((NUM_TERR_CELLS, W), dtype=int)
    for m in range(NUM_TERR_CELLS):
        for s in range(NUM_VISIBLE_SATS):
            N_s_m = set(np.where(O[s, :, m] == 1)[0])
            for t in range(W):
                served = set(bh_patterns[s][t])
                if served & N_s_m:
                    interference_count[m, t] += 1

    return {
        'O': O, 'bh_patterns': bh_patterns, 'delays': delays,
        'terr_pos': terr_pos, 'sat_cell_pos': sat_cell_pos,
        'disturbed_terr_idx': disturbed_terr_idx,
        'disturbed_sat_idx': disturbed_sat_idx,
        'interference_count': interference_count,
    }


def time_alignment_proposed(interference_count, T, num_terr, W_len):
    """
    Proposed 方法 (论文 Algorithm 1):
    - 干扰时隙内部: 完全不可用
    - 干扰边界时隙: 通过特殊时隙配置部分可用 (ST_c2s / ST_s2c 个符号)
    - 非干扰时隙: 完全可用
    返回 gamma[num_terr]
    """
    # 特殊时隙可用符号数 (由 Eq.10 计算, 典型值 3-5 个)
    ST_C2S = 3  # BS→卫星方向边界可用符号数
    ST_S2C = 3  # 卫星→BS方向边界可用符号数

    gamma = np.zeros(num_terr)
    for m in range(num_terr):
        usable = 0
        for t in range(W_len):
            if interference_count[m, t] == 0:
                usable += N_SYM  # 完全可用
            else:
                # 判断是否为干扰期的边界时隙
                prev_int = interference_count[m, t-1] > 0 if t > 0 else False
                next_int = interference_count[m, t+1] > 0 if t < W_len-1 else False
                is_start = not prev_int  # 干扰期起始
                is_end = not next_int    # 干扰期结束

                if is_start and is_end:
                    # 单时隙干扰期: 两个边界合并 → 用较少的符号
                    usable += max(ST_C2S, ST_S2C)
                elif is_start:
                    usable += ST_C2S
                elif is_end:
                    usable += ST_S2C
                # else: 内部时隙, 完全不可用 (0 symbols)

        gamma[m] = usable / (W_len * N_SYM)
    return gamma


def time_alignment_timeslot_based(interference_count, T, num_terr, W_len):
    """
    Timeslot-based: 不使用特殊时隙, 边界时隙完全浪费
    内部干扰时隙也不可用
    """
    gamma = np.zeros(num_terr)
    for m in range(num_terr):
        n_available = 0
        for t in range(W_len):
            if interference_count[m, t] == 0:
                # 还要检查相邻时隙是否干扰 (边界浪费)
                prev_int = interference_count[m, t-1] > 0 if t > 0 else False
                next_int = interference_count[m, t+1] > 0 if t < W_len-1 else False
                if prev_int or next_int:
                    n_available += 0  # 边界时隙完全浪费
                else:
                    n_available += 1
        gamma[m] = n_available / W_len
    return gamma


def time_alignment_general_sync(interference_count, T, num_terr, W_len):
    """
    General time sync: 每次干扰事件前需要同步
    干扰时隙不可用 + 同步开销
    """
    T_sync = 300e-6  # 同步时间
    sync_loss = T_sync / T  # 同步开销占一个时隙的比例

    gamma = np.zeros(num_terr)
    for m in range(num_terr):
        n_sync_events = 0
        n_available = 0
        for t in range(W_len):
            if interference_count[m, t] == 0:
                prev_int = interference_count[m, t-1] > 0 if t > 0 else False
                if prev_int:
                    n_available += max(0, 1 - sync_loss)  # 同步后恢复
                    n_sync_events += 1
                else:
                    n_available += 1
        gamma[m] = n_available / W_len
    return gamma
