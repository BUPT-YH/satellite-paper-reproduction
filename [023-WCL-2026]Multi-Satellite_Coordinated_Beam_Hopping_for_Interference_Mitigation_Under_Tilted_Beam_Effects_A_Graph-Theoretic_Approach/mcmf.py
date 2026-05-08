"""
最小费用最大流 (MCMF) 算法
使用 Successive Shortest Path (SPFA) 求解
用于生成高质量的初始卫星-小区关联方案
"""

import numpy as np
from collections import deque


class MCMFGraph:
    """最小费用最大流图"""

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.edges = []       # 边列表: [to, cap, cost, rev_idx]
        self.adj = [[] for _ in range(n_nodes)]

    def add_edge(self, u, v, cap, cost):
        """添加边 u→v，容量cap，单位费用cost"""
        self.adj[u].append(len(self.edges))
        self.edges.append([v, cap, cost, len(self.edges) + 1])
        self.adj[v].append(len(self.edges))
        self.edges.append([u, 0, -cost, len(self.edges) - 1])

    def spfa(self, src, sink):
        """SPFA求最短增广路径（支持负权边）"""
        dist = np.full(self.n, np.inf)
        in_queue = np.zeros(self.n, dtype=bool)
        prev_edge = np.full(self.n, -1, dtype=int)
        dist[src] = 0
        in_queue[src] = True
        q = deque([src])

        while q:
            u = q.popleft()
            in_queue[u] = False
            for ei in self.adj[u]:
                e = self.edges[ei]
                v, cap, cost = e[0], e[1], e[2]
                if cap > 0 and dist[u] + cost < dist[v]:
                    dist[v] = dist[u] + cost
                    prev_edge[v] = ei
                    if not in_queue[v]:
                        in_queue[v] = True
                        q.append(v)

        if dist[sink] == np.inf:
            return None, None

        # 回溯增广路径
        path = []
        v = sink
        while v != src:
            ei = prev_edge[v]
            path.append(ei)
            v = self.edges[ei ^ 1][0]  # 反向边的起点

        return dist[sink], path

    def solve(self, src, sink):
        """求解MCMF，返回 (最大流, 最小费用, 流分配)"""
        total_flow = 0
        total_cost = 0.0

        while True:
            cost, path = self.spfa(src, sink)
            if path is None:
                break

            # 确定增广量
            flow = np.inf
            for ei in path:
                flow = min(flow, self.edges[ei][1])

            # 更新残差网络
            for ei in path:
                self.edges[ei][1] -= flow
                self.edges[ei ^ 1][1] += flow

            total_flow += flow
            total_cost += flow * cost

        return total_flow, total_cost


def solve_mcmf_sca(elev, Ls, T, delta_L=30):
    """
    用MCMF求解初始卫星-小区关联(SCA)方案

    elev: (S, C) 仰角矩阵（弧度）
    Ls: 每颗卫星最大同时激活波束数
    T: BH周期数
    delta_L: MCMF余量

    返回: s_assign (C,) 每个小区的服务卫星索引
    """
    S, C = elev.shape
    # 节点: src=0, sink=1, satellites=2..S+1, cells=S+2..S+C+1
    n_nodes = 2 + S + C
    src = 0
    sink = 1

    g = MCMFGraph(n_nodes)

    # 源到卫星的边：容量 = Ls*T - delta_L，费用=0
    for s in range(S):
        cap = max(Ls * T - delta_L, 1)
        g.add_edge(src, 2 + s, cap, 0)

    # 卫星到小区的边：容量=1，费用=-仰角（最大化仰角和）
    for s in range(S):
        for c in range(C):
            cost = -elev[s, c]  # 负费用 = 最大化仰角
            g.add_edge(2 + s, 2 + S + c, 1, cost)

    # 小区到汇的边：容量=1，费用=0
    for c in range(C):
        g.add_edge(2 + S + c, sink, 1, 0)

    total_flow, total_cost = g.solve(src, sink)

    # 从流中提取SCA方案
    s_assign = np.zeros(C, dtype=int)
    for s in range(S):
        for ei in g.adj[2 + s]:
            e = g.edges[ei]
            if e[0] >= 2 + S and e[1] == 0 and ei % 2 == 0:
                # 原始边流量用完（cap=0），说明有流通过
                c = e[0] - 2 - S
                s_assign[c] = s

    return s_assign
