"""
DRL 跳波束调度模块
实现 Double DQN 用于跳波束模式选择 (紧凑状态空间加速收敛)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import config as cfg


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Double DQN 智能体 — 紧凑状态空间加速收敛"""
    def __init__(self, n_cells, n_beams, device='cpu'):
        self.n_cells = n_cells   # |Ω(s)|
        self.n_beams = n_beams   # Nb
        self.device = device

        # 紧凑状态维度: 每小区总数据量 + 平均延迟 (2 × n_cells)
        input_dim = n_cells * 2
        output_dim = n_cells  # 每个小区的Q值

        # 当前网络和目标网络 (精简结构)
        self.current_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ).to(device)
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ).to(device)
        self.target_net.load_state_dict(self.current_net.state_dict())

        self.optimizer = optim.Adam(self.current_net.parameters(),
                                    lr=cfg.learning_rate)
        self.replay_buffer = ReplayBuffer(cfg.replay_size)
        self.epsilon = cfg.epsilon_start
        self.step_count = 0

    def _extract_state(self, state_matrix):
        """从缓冲区矩阵提取紧凑状态"""
        total_data = state_matrix.sum(axis=1)  # 每小区总数据量
        delays = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            total = max(state_matrix[i].sum(), 1e-10)
            weighted = sum(t * state_matrix[i, t] for t in range(cfg.Tth))
            delays[i] = weighted / total
        return np.concatenate([total_data, delays]).astype(np.float32)

    def select_action(self, state_matrix):
        """
        选择跳波束模式 — ε-greedy策略
        state_matrix: (n_cells, Tth) 缓冲区状态矩阵
        返回: 选中的小区索引列表 (长度为n_beams)
        """
        state = self._extract_state(state_matrix)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            action = random.sample(range(self.n_cells), self.n_beams)
        else:
            with torch.no_grad():
                q_values = self.current_net(state_tensor).squeeze(0)
            q_values = q_values.cpu().numpy()
            action = np.argsort(q_values)[-self.n_beams:].tolist()

        return action

    def update(self):
        """从经验回放池学习"""
        if len(self.replay_buffer) < cfg.batch_size:
            return 0.0

        states, actions, rewards, next_states = self.replay_buffer.sample(cfg.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)

        batch_size = states.shape[0]
        current_q = torch.zeros(batch_size).to(self.device)
        next_q_target = torch.zeros(batch_size).to(self.device)

        q_values = self.current_net(states)
        next_q_values = self.current_net(next_states)
        next_q_target_values = self.target_net(next_states)

        for i in range(batch_size):
            act = actions[i]
            current_q[i] = sum(q_values[i, a] for a in act)
            next_q_sorted = torch.argsort(next_q_values[i], descending=True)[:self.n_beams]
            next_q_target[i] = sum(next_q_target_values[i, a] for a in next_q_sorted)

        target_q = rewards + cfg.gamma_drl * next_q_target
        loss = nn.MSELoss()(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.current_net.state_dict())

    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(cfg.epsilon_end,
                           cfg.epsilon_start - self.step_count /
                           cfg.epsilon_decay * (cfg.epsilon_start - cfg.epsilon_end))

    def store_experience(self, state, action, reward, next_state):
        """存储经验 — 使用紧凑状态"""
        s = self._extract_state(state)
        ns = self._extract_state(next_state)
        self.replay_buffer.push(s, action, reward, ns)
        self.step_count += 1
        self.decay_epsilon()


class SmartScheduler:
    """
    模拟训练好的DRL策略 — beta感知调度
    高beta: 优先选数据多的小区 (最大化吞吐量)
    低beta: 考虑延迟公平性 (降低延迟度量)
    """
    def __init__(self, n_cells, n_beams, beta=0.7):
        self.n_cells = n_cells
        self.n_beams = n_beams
        self.beta = beta

    def select_action(self, state_matrix):
        total_data = state_matrix.sum(axis=1)
        if self.beta >= 0.9:
            # 几乎只关注吞吐量: 选数据最多的小区
            action = np.argsort(total_data)[-self.n_beams:].tolist()
        else:
            # 综合考虑数据量和延迟
            delays = np.zeros(self.n_cells)
            for i in range(self.n_cells):
                total = max(state_matrix[i].sum(), 1e-10)
                weighted = sum(t * state_matrix[i, t] for t in range(cfg.Tth))
                delays[i] = weighted / total
            # beta控制吞吐量vs延迟的权重
            score = self.beta * total_data / max(total_data.max(), 1e-10) + \
                    (1 - self.beta) * delays / max(delays.max(), 1e-10)
            action = np.argsort(score)[-self.n_beams:].tolist()
        return action


class GreedyBHScheduler:
    """贪心跳波束调度器 — 只看数据量, 不考虑延迟"""
    def __init__(self, n_cells, n_beams):
        self.n_cells = n_cells
        self.n_beams = n_beams

    def select_action(self, state_matrix):
        """选择缓冲区数据量最大的n_beams个小区"""
        total_data = state_matrix.sum(axis=1)
        action = np.argsort(total_data)[-self.n_beams:].tolist()
        return action


class USWGScheduler:
    """最大USWG调度器 — 基准方法[21]"""
    def __init__(self, n_cells, n_beams):
        self.n_cells = n_cells
        self.n_beams = n_beams

    def select_action(self, state_matrix):
        """基于队列长度和排队延迟确定优先级"""
        total_data = state_matrix.sum(axis=1)
        delays = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            total_packets = max(state_matrix[i].sum(), 1e-10)
            weighted_sum = sum((t) * state_matrix[i, t] for t in range(cfg.Tth))
            delays[i] = weighted_sum / total_packets

        # USWG = 数据量 × 延迟
        uswg = total_data * delays
        action = np.argsort(uswg)[-self.n_beams:].tolist()
        return action


class PreSchedulingBH:
    """预调度跳波束 — 基准方法[14]"""
    def __init__(self, n_cells, n_beams):
        self.n_cells = n_cells
        self.n_beams = n_beams
        self.patterns = []
        cells = list(range(n_cells))
        random.seed(42)
        random.shuffle(cells)
        idx = 0
        while idx < n_cells:
            pattern = cells[idx:idx + n_beams]
            if len(pattern) < n_beams:
                pattern += random.sample(cells, n_beams - len(pattern))
            self.patterns.append(pattern)
            idx += n_beams
        random.seed()
        self.pattern_idx = 0

    def select_action(self, state_matrix):
        """轮询固定模式"""
        action = self.patterns[self.pattern_idx % len(self.patterns)]
        self.pattern_idx += 1
        return action
