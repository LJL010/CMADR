# AC.py - 修正版：提高数值稳定性，符合论文要求

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()

        # === 修正：增加网络深度，符合论文复杂度要求 ===
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)  # 增加一层，逐渐减少维度
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)

        # === 改进1：更好的权重初始化 ===
        # 使用He初始化，适合ReLU激活函数
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        # 输出层使用Xavier初始化
        nn.init.xavier_uniform_(self.fc4.weight)

        # 偏置初始化
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)
        nn.init.constant_(self.fc4.bias, 0.0)  # 输出层偏置为0

        # === 改进2：添加Batch Normalization ===
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # === 改进3：添加Dropout ===
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, obs):
        batch_size = obs.shape[0]

        # === 改进4：更稳定的前向传播 ===
        x = F.relu(self.fc1(obs))
        if batch_size > 1:  # BatchNorm需要batch_size > 1
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if batch_size > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        if batch_size > 1:
            x = self.bn3(x)
        x = self.dropout(x)

        # === 改进5：输出logits而不是直接概率 ===
        logits = self.fc4(x)

        # === 修正：根据论文要求，输出稳定的概率分布 ===
        # 限制logits范围，避免数值溢出
        logits = torch.clamp(logits, min=-10, max=10)

        # 使用温度参数软化概率分布，提高探索性
        temperature = 1.0
        logits = logits / temperature

        # 计算稳定的概率分布
        probs = F.softmax(logits, dim=-1)

        # === 改进6：确保概率不为0，避免log(0) ===
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs

    def get_logits(self, obs):
        """
        专门用于训练时获取logits的方法
        """
        batch_size = obs.shape[0]

        x = F.relu(self.fc1(obs))
        if batch_size > 1:
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if batch_size > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        if batch_size > 1:
            x = self.bn3(x)
        x = self.dropout(x)

        logits = self.fc4(x)
        logits = torch.clamp(logits, min=-10, max=10)

        return logits


class CriticNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()

        # === 修正：增加网络深度，与Actor保持一致 ===
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)

        # 权重初始化
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc4.weight)

        # 偏置初始化
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        nn.init.constant_(self.fc3.bias, 0.01)
        nn.init.constant_(self.fc4.bias, 0.0)

        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, obs):
        batch_size = obs.shape[0]

        x = F.relu(self.fc1(obs))
        if batch_size > 1:
            x = self.bn1(x)
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        if batch_size > 1:
            x = self.bn2(x)
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        if batch_size > 1:
            x = self.bn3(x)
        x = self.dropout(x)

        value = self.fc4(x)
        return value


class AdvancedActorNet(nn.Module):
    """
    高级版Actor网络，专门为大规模ISTN网络设计
    """

    def __init__(self, obs_dim, hidden_dim, action_dim, use_attention=False):
        super().__init__()

        self.use_attention = use_attention

        # 编码层
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
        )

        # === 注意力机制（可选，用于大规模网络） ===
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # 决策层
        self.decision_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim // 2, action_dim),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, obs):
        batch_size = obs.shape[0]

        # 编码
        x = self.encoder(obs)

        # 注意力机制（如果启用）
        if self.use_attention and batch_size > 1:
            x = x.unsqueeze(1)  # [batch, 1, hidden]
            attended, _ = self.attention(x, x, x)
            x = self.attention_norm(x + attended)
            x = x.squeeze(1)  # [batch, hidden]

        # 决策
        logits = self.decision_net(x)
        logits = torch.clamp(logits, min=-10, max=10)

        # 概率分布
        probs = F.softmax(logits, dim=-1)
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs