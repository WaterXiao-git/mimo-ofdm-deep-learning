# 改进的无线通信系统深度学习模型

## 项目概述

这是一个基于深度学习的端到端无线通信系统，实现了上行链路反馈和下行链路数据传输的联合优化。

## 主要改进

### 1. 模型架构改进
- **自适应方案选择**: 根据信道条件和SNR自动选择最优传输方案
- **注意力机制**: 信道注意力模块，自适应关注重要的信道特征
- **SNR感知处理**: 根据信噪比调整信号处理策略
- **残差连接**: 提高模型训练稳定性和收敛速度
- **预编码优化**: 改进的发射机预编码策略
- **软判决解码**: 更精确的接收机软判决网络

### 2. 训练策略改进
- **Focal Loss**: 处理数据不平衡问题
- **学习率调度**: 余弦退火学习率调度
- **梯度裁剪**: 防止梯度爆炸
- **早停机制**: 防止过拟合
- **批量大小优化**: 从2增加到32，提高训练稳定性

### 3. 评估系统改进
- **全面性能评估**: 多维度性能指标
- **SNR敏感性分析**: 不同信噪比下的性能曲线
- **可视化结果**: 自动生成性能图表
- **统计分析**: 均值、标准差等统计信息

## 文件结构

```
├── modelDesign_improved.py    # 改进的模型架构
├── modelTrain_improved.py     # 改进的训练脚本
├── modelEval_improved.py      # 改进的评估脚本
├── config.py                  # 配置文件
├── README_improved.md         # 使用说明
├── modelSubmit/              # 模型保存目录
├── logs/                     # 训练日志
└── evaluation_results/       # 评估结果
```

## 使用方法

### 1. 环境准备
```bash
pip install torch numpy matplotlib tqdm einops
```

### 2. 数据准备
确保训练数据 `H_train.npy` 位于 `./data_train/` 目录下。

### 3. 训练模型
```bash
python modelTrain_improved.py
```

训练过程会：
- 自动创建必要的目录
- 显示训练进度和性能指标
- 保存最佳模型和训练历史
- 生成训练曲线图

### 4. 评估模型
```bash
python modelEval_improved.py
```

评估过程会：
- 加载训练好的模型
- 进行全面的性能评估
- 分析不同SNR下的性能
- 生成详细的评估报告和图表

## 核心技术特性

### 信道注意力机制
```python
class ChannelAttention(nn.Module):
    def __init__(self, input_dim, reduction=16):
        # 自适应关注重要的信道特征
```

### SNR感知处理
```python
class SNRAwareBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # 根据SNR调整处理策略
```

### 自适应方案选择
```python
# 根据信道条件自动选择最优方案
scheme_weights = self.scheme_selector(features)
output = weighted_combination(scheme1, scheme2, scheme3, scheme_weights)
```

## 性能指标

模型评估包括以下指标：
- **总体得分**: 综合性能评分
- **误码率(BER)**: 不同SNR下的误码率
- **吞吐量**: 成功传输的数据比例
- **SNR敏感性**: 不同信噪比下的性能变化

## 配置参数

所有参数都集中在 `config.py` 中管理：
- 系统参数：天线数量、子载波数等
- 模型参数：隐藏层维度、dropout率等
- 训练参数：学习率、批量大小等
- 评估参数：测试次数、SNR范围等

## 输出结果

### 训练输出
- `./modelSubmit/`: 训练好的模型文件
- `./logs/`: 训练损失和准确率历史
- `./logs/training_curves.png`: 训练曲线图

### 评估输出
- `./evaluation_results/performance_curves.png`: 性能曲线图
- `./evaluation_results/evaluation_results.npy`: 详细评估数据
- 控制台输出：各SNR下的性能统计

## 主要改进效果

1. **收敛速度**: 通过注意力机制和残差连接，模型收敛更快
2. **泛化能力**: SNR感知处理提高了不同信道条件下的适应性
3. **训练稳定性**: 改进的损失函数和训练策略提高了稳定性
4. **评估全面性**: 多维度评估提供了更全面的性能分析

## 注意事项

1. 确保有足够的GPU内存（推荐8GB以上）
2. 训练时间较长，建议使用GPU加速
3. 可根据实际需求调整 `config.py` 中的参数
4. 如遇到内存不足，可适当减小批量大小

## 扩展建议

1. **多GPU训练**: 支持分布式训练
2. **模型压缩**: 知识蒸馏和模型剪枝
3. **在线学习**: 支持增量学习和适应
4. **硬件优化**: 针对特定硬件的优化