# CMADR - 修正版本
This is an implementation of CMADR algorithm strictly following the paper.

**Paper**: [Dynamic Routing for Integrated Satellite-Terrestrial Networks: A Constrained Multi-Agent Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/10436098/authors#authors)

## 🚀 最新修正

### 主要改进
- ✅ **完整的网络架构**：分离的Global Reward Critic + Global Cost Critic
- ✅ **严格的约束处理**：三类拉格朗日乘子分别处理对应约束
- ✅ **论文标准奖励**：平均传输速率定义
- ✅ **数值稳定性**：改进的Actor-Critic网络
- ✅ **维度一致性**：统一的观测空间处理

### 关键修正
1. **MASys.py**: 增加分离的Global Reward/Cost Critics
2. **LM.py**: 完全重写训练逻辑，严格按照论文公式21-31
3. **AC.py**: 增强网络稳定性，添加BatchNorm和Dropout
4. **ISTN_ENV.py**: 修正奖励函数和观测维度计算
5. **train.py**: 适应新架构，动态计算动作维度

## 🏗️ 架构概览

### 网络结构
```
每个Agent:
├── Actor Network (策略网络)
└── Local Critic Network (本地价值网络)

全局网络:
├── Global Reward Critic (全局奖励价值网络)
└── Global Cost Critic (全局成本价值网络)

约束处理:
├── λC: 全局丢包率约束 (公式12)
├── λj: 地面站能耗约束 (公式13)
└── λi: 卫星能耗约束 (公式14)
```

### 观测空间
- **卫星**: 自身状态(3) + 4个邻居状态(12) + 到目的地距离(1) = 16维
- **地面站**: 自身状态(1) + 所有卫星状态(2×N_sat) + 到目的地距离(1)

### 动作空间
- **卫星**: 选择下一跳邻居 (4个ISL邻居 + 可达地面站)
- **地面站**: 选择上传卫星 (可达卫星列表)

## 📖 使用方法

### 1. 训练

```bash
python train.py --config config.json
```

**配置参数**:
```json
{
  "train": {
    "num_episodes": 300,
    "gamma": 0.98,
    "cost_limits": {
      "energy": 10.0,    // 能耗约束 (KJ)
      "loss": 0.01       // 丢包率约束 (1%)
    },
    "batch_size": 50,
    "learning_rates": {
      "actor": 1e-3,
      "critic": 1e-3,
      "lagrange": 0.01
    }
  }
}
```

### 2. 预测

```bash
python predict.py --config config.json
```

### 3. 训练日志

训练过程会生成详细日志：
```
training_logs/
├── training_YYYYMMDD_HHMMSS/
│   ├── episode_0000.log        # 每轮详细日志
│   ├── episode_0001.log
│   ├── training_summary.log    # 训练摘要
│   └── final_summary.log       # 最终总结
```

## 📊 性能指标

### 评估指标
- **平均延迟**: 端到端数据包传输延迟
- **丢包率**: 由于缓冲区溢出导致的丢包比例
- **能耗**: 卫星和地面站的总能量消耗
- **传输效率**: 成功交付的数据包数量

### 约束监控
- **能耗约束**: 每个节点的累积能耗 ≤ 阈值
- **丢包约束**: 全局丢包率 ≤ 阈值
- **拉格朗日乘子**: 约束违反程度的指示器

## 🔧 数据格式

### 输入数据
项目的数据位于 `data/` 目录下，每个文件都是一个 JSON，主要字段说明如下：

```json
{
    "num_satellites": 1462,
    "num_ground": 261,
    "num_train_slots": 574,
    "num_predict_slots": 574,
    "sat_positions_per_slot": [...],  // 卫星轨道数据
    "gs_positions": [...],            // 地面站位置
    "train_queries": [...],           // 训练期间的通信请求
    "predict_queries": [...]          // 预测期间的通信请求
}
```

### 查询格式
```json
{
    "src": 1,     // 源地面站ID
    "dst": 4,     // 目标地面站ID  
    "time": 7     // 时隙
}
```

## ⚙️ 核心算法

### CMADR算法流程
1. **观测收集**: 每个agent收集本地观测
2. **动作选择**: 使用Actor网络选择邻居
3. **环境交互**: 执行动作，获得奖励和成本
4. **价值估计**: 使用Critic网络估计状态价值
5. **策略更新**: 使用PPO更新Actor参数
6. **约束处理**: 使用拉格朗日乘子处理约束
7. **全局优化**: 更新Global Critics

### 关键公式
- **Actor损失** (公式21): `L_actor = -E[log π(a|s) * A] + λ * C`
- **Critic损失** (公式28): `L_critic = MSE(V(s), target)`
- **拉格朗日更新** (公式26-27): `λ^(k+1) = ReLU(λ^k + α * ∇C)`

## 🔍 调试指南

### 常见问题
1. **维度不匹配**: 检查obs_dim和action_dim计算
2. **概率为0**: Actor网络输出概率过小，检查数值稳定性
3. **约束违反**: 拉格朗日乘子学习率过大，调整α_λ
4. **收敛缓慢**: 批处理大小过小，增加batch_size

### 性能调优
- **网络深度**: 根据数据规模调整hidden_dim
- **学习率**: Actor和Critic使用不同学习率
- **约束权重**: 调整cost_limits以平衡性能和约束
- **探索策略**: 使用ε-greedy或温度参数

## 📚 项目结构

```
CMADR/
├── AC.py              # Actor-Critic网络（增强版）
├── ISTN_ENV.py        # 环境定义（修正版）
├── LM.py              # 拉格朗日训练逻辑（重写版）
├── MASys.py           # 多智能体系统（分离Critics版）
├── train.py           # 训练脚本（适应新架构）
├── predict.py         # 预测脚本（执行模式）
├── config.json        # 配置文件（更新参数）
├── data/              # 数据集目录
├── model/             # 训练模型保存
├── training_logs/     # 训练日志
└── README.md          # 项目说明（本文件）
```

## 🎯 与论文对比

### 严格遵循论文要求
- ✅ **Figure 3架构**: 完整的Actor-Critic + Global Critics
- ✅ **公式21-31**: 严格按照数学公式实现
- ✅ **Figure 2观测**: 卫星和地面站观测空间
- ✅ **Table II参数**: 使用论文中的超参数设置
- ✅ **约束处理**: 三类拉格朗日约束分别处理

### 预期性能
根据论文Figure 4-5，CMADR应该能够：
- 减少21%+的平均延迟（相比基线算法）
- 满足能耗和丢包率约束
- 在大规模卫星网络中稳定运行

## 🚨 注意事项

1. **模型兼容性**: 新架构与旧模型不兼容，需重新训练
2. **计算资源**: 分离的Critics增加内存和计算需求  
3. **数据质量**: 确保neighbor数据和查询数据格式正确
4. **约束调优**: 过严的约束可能导致性能下降

## 📞 技术支持

如果在使用过程中遇到问题：
1. 检查日志文件中的详细错误信息
2. 验证数据格式和配置参数
3. 参考论文的实验设置和参数
4. 考虑使用更小的数据集进行调试

---

**版本**: CMADR v2.0 (严格论文实现版)
**最后更新**: 2025年
**论文引用**: Lyu et al., "Dynamic Routing for Integrated Satellite-Terrestrial Networks: A Constrained Multi-Agent Reinforcement Learning Approach," IEEE Journal on Selected Areas in Communications, 2024.