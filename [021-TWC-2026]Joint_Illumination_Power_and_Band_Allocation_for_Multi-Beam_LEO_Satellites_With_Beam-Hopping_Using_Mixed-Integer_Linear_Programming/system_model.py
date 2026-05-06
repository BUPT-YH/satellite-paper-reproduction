"""
系统模型：六边形小区布局、用户分布、邻接矩阵、链路预算
"""
import numpy as np
from config import N_ILL, N_FREQ, M_NUM, DU_MAX


def generate_hex_cells(n_rings):
    """
    生成六边形小区布局（环状）
    n_rings=2 → 19 cells, n_rings=3 → 37 cells, n_rings=9 → 271 cells
    使用 axial 坐标系
    """
    cells = [(0, 0)]
    for ring in range(1, n_rings + 1):
        x, y = 0, ring
        for _ in range(ring):
            cells.append((x, y))
            x += 1
        for _ in range(ring):
            cells.append((x, y))
            y -= 1
        for _ in range(ring):
            cells.append((x, y))
            x -= 1
        for _ in range(ring):
            cells.append((x, y))
            y -= 1
        for _ in range(ring):
            cells.append((x, y))
            x -= 1
        for _ in range(ring):
            cells.append((x, y))
            y += 1
    return cells


def cell_distance(c1, c2):
    """六边形 axial 坐标的距离"""
    q1, r1 = c1
    q2, r2 = c2
    return (abs(q1 - q2) + abs(q1 + r1 - q2 - r2) + abs(r1 - r2)) // 2


def compute_adjacency(cells):
    """计算小区邻接矩阵（距离为1的小区互为邻接）"""
    nc = len(cells)
    adj = np.zeros((nc, nc), dtype=int)
    for i in range(nc):
        for j in range(i + 1, nc):
            if cell_distance(cells[i], cells[j]) == 1:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def get_n_rings(nc):
    """根据小区数计算环数"""
    # 3*n*(n-1)+1 = nc → n = (1+sqrt(1-4*(1-nc)/3))/2
    n = int(np.ceil((1 + np.sqrt(1 + 4 * (nc - 1) / 3)) / 2))
    # 验证
    actual = 3 * n * (n - 1) + 1
    while actual < nc:
        n += 1
        actual = 3 * n * (n - 1) + 1
    return n


class BeamHoppingSystem:
    """跳波束系统模型"""

    def __init__(self, scenario, seed=42):
        self.scenario = scenario
        self.NC = scenario["NC"]
        self.NT = scenario["NT"]
        self.GTX = scenario["GTX"]

        # 生成小区布局
        n_rings = get_n_rings(self.NC)
        self.cells = generate_hex_cells(n_rings)[:self.NC]
        self.adj_c = compute_adjacency(self.cells)

        # 随机生成用户
        rng = np.random.RandomState(seed)
        self.nu = scenario["NU"]
        self.seed = seed
        self.rng = rng

        # 用户位置映射到小区（均匀分布）
        self.user_to_cell = rng.randint(0, self.NC, size=self.nu)

        # 用户-小区关联矩阵 UpC
        self.upc = np.zeros((self.nu, self.NC), dtype=int)
        for u in range(self.nu):
            self.upc[u, self.user_to_cell[u]] = 1

        # 用户邻接矩阵 AdjU = UpC @ AdjC @ UpC^T → (NU, NU)
        self.adj_u = self.upc @ self.adj_c @ self.upc.T
        # 共小区用户也视为"邻接"
        for u in range(self.nu):
            for v in range(u + 1, self.nu):
                if self.user_to_cell[u] == self.user_to_cell[v]:
                    self.adj_u[u, v] = 1
                    self.adj_u[v, u] = 1

        # 用户需求 (Mbps), 均匀分布 U(0, DU_MAX)
        self.demand = rng.uniform(0, DU_MAX, size=self.nu)

        # 每个时隙的需求 (每个时隙需求不变，因为是固定快照)
        self.R = np.tile(self.demand, (self.NT, 1)).T  # shape: (NU, NT)

        # 预计算功率和容量
        from config import compute_power_and_capacity
        self.P_req, self.D = compute_power_and_capacity(self.GTX)

    def get_cell_demand(self, c, t=None):
        """获取小区 c 在时刻 t 的总需求"""
        users_in_cell = np.where(self.upc[:, c] == 1)[0]
        if t is not None:
            return sum(self.R[u, t] for u in users_in_cell)
        return sum(self.demand[u] for u in users_in_cell)

    def get_adjacent_pairs(self):
        """获取邻接用户对（用于频率约束）"""
        pairs = []
        for u in range(self.nu):
            for v in range(u + 1, self.nu):
                if self.adj_u[u, v] == 1:
                    pairs.append((u, v))
        return pairs
