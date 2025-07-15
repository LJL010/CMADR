# AC.py - 改进Actor网络的数值稳定性

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 增加一层
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # === 改进1：权重初始化 ===
        # 使用Xavier初始化，提高数值稳定性
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        # 偏置初始化为小的正值
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)

    def forward(self, obs):
        # === 改进2：增加批归一化和dropout ===
        x = F.relu(self.fc1(obs))
        x = F.dropout(x, p=0.1, training=self.training)  # 轻微dropout

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1, training=self.training)

        # === 改进3：更稳定的softmax ===
        logits = self.fc3(x)

        # 限制logits范围，避免数值溢出
        logits = torch.clamp(logits, min=-10, max=10)

        # 使用数值稳定的softmax
        probs = F.softmax(logits, dim=-1)

        # === 改进4：确保概率不为0 ===
        # 添加小的常数，避免概率为0
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs


class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 增加一层
        self.fc3 = nn.Linear(hidden_dim, 1)

        # 权重初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.dropout(x, p=0.1, training=self.training)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.1, training=self.training)

        value = self.fc3(x)
        return value