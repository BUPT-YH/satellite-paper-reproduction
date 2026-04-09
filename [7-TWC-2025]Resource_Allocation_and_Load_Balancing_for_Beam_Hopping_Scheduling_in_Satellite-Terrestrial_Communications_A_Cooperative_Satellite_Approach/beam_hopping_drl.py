"""
DRL 跳波束调度模块
实现 Double DQN 用于跳波束模式选择
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

import config as cfg


class DQNNetwork(nn.Module):
    """DQN网络 — 6层全连接网络"""
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


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
    """Double DQN 智能体 — 每颗卫星一个"""
    def __init__(self, n_cells, n_beams, device='cpu'):
        self.n_cells = n_cells   # |Ω(s)|
        self.n_beams = n_beams   # Nb
        self.device = device

        # 状态维度: |Ω(s)| × Tth (缓冲区状态矩阵展平)
        input_dim = n_cells * cfg.Tth
        output_dim = n_cells  # 每个小区的Q值

        # 当前网络和目标网络
        self.current_net = DQNNetwork(input_dim, output_dim).to(device)
        self.target_net = DQNNetwork(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.current_net.state_dict())

        self.optimizer = optim.Adam(self.current_net.parameters(),
                                    lr=cfg.learning_rate)
        self.replay_buffer = ReplayBuffer(cfg.replay_size)
        self.epsilon = cfg.epsilon_start
        self.step_count = 0

    def select_action(self, state_matrix):
        """
        选择跳波束模式 — ε-greedy策略
        state_matrix: (n_cells, Tth) 缓冲区状态矩阵
        返回: 选中的小区索引列表 (长度为n_beams)
        """
        state_flat = state_matrix.flatten().astype(np.float32)
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)

        if random.random() < self.epsilon:
            # 探索: 随机选择n_beams个不重复的小区
            action = random.sample(range(self.n_cells), self.n_beams)
        else:
            # 利用: 选择Q值最大的n_beams个小区
            with torch.no_grad():
                q_values = self.current_net(state_tensor).squeeze(0)
            q_values = q_values.cpu().numpy()
            # 选择Q值最高的n_beams个小区
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

        # 当前Q值: 对action中选中的小区取Q值之和
        # actions: (batch_size, n_beams) 每行是选中的小区索引
        batch_size = states.shape[0]
        current_q = torch.zeros(batch_size).to(self.device)
        next_q_current = torch.zeros(batch_size).to(self.device)
        next_q_target = torch.zeros(batch_size).to(self.device)

        q_values = self.current_net(states)        # (B, n_cells)
        next_q_values = self.current_net(next_states)  # (B, n_cells)
        next_q_target_values = self.target_net(next_states)  # (B, n_cells)

        for i in range(batch_size):
            act = actions[i]
            # Q(Ds(n), as(n)) = Σ_k Q̂(Ds(n), as,k(n))
            current_q[i] = sum(q_values[i, a] for a in act)
            # Double DQN: 用当前网络选择动作，目标网络评估
            next_q_sorted = torch.argsort(next_q_values[i], descending=True)[:self.n_beams]
            next_q_current[i] = sum(next_q_values[i, a] for a in next_q_sorted)
            next_q_target[i] = sum(next_q_target_values[i, a] for a in next_q_sorted)

        # 目标Q值
        target_q = rewards + cfg.gamma_drl * next_q_target
        # 计算loss
        loss = nn.MSELoss()(current_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
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
        """存储经验"""
        self.replay_buffer.push(state.flatten(), action, reward, next_state.flatten())
        self.step_count += 1
        self.decay_epsilon()


class GreedyBHScheduler:
    """贪心跳波束调度器 — 基准方法"""
    def __init__(self, n_cells, n_beams):
        self.n_cells = n_cells
        self.n_beams = n_beams

    def select_action(self, state_matrix):
        """选择缓冲区数据量最大的n_beams个小区"""
        # state_matrix: (n_cells, Tth)
        total_data = state_matrix.sum(axis=1)  # 各小区总数据量
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
        # 计算排队延迟
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
        # 生成固定的预调度模式
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
