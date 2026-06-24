# MIMO-OFDM Deep Learning — 端到端深度学习无线通信系统

> **基于深度学习的端到端 MIMO-OFDM 无线通信系统**。
> 设计智能编码器（Transformer-based）、预编码发射机和信号检测接收机，联合训练取代传统分模块方案。

---

## 📋 系统参数 | System Specs

| 参数 | 数值 | 说明 |
|------|------|------|
| 天线配置 | 32 TX × 4 RX | 大规模 MIMO |
| 上行子载波 | 96 | 信道反馈 |
| 下行子载波 | 144 | 数据传输 |
| 信噪比范围 | -20 ~ +20 dB | 适应极端条件 |
| 损失函数 | Focal Loss (α=1.0, γ=2.0) | 处理类别不平衡 |

## 💡 三大学习模块 | 3 Learned Modules

### 1. Encoder（上行信道编码器）
- 特征值分解 → Transformer 编码 → 96 个压缩符号反馈

### 2. Transmitter（下行发射机）
- 反馈重建 → 可学习调制层 → 32 天线并行发射

### 3. Receiver（接收机/检测器）
- 10 个 Conv3×3 残差块 → 对数似然比（LLR）输出

## 🚀 快速开始 | Quick Start

```bash
# PyTorch 版
cd pytorch
python modelTrain_optimized.py

# TensorFlow 版
cd ../tensorflow
python modelTrain.py
```

## 📁 项目结构 | Project Structure

```
├── pytorch/                  # PyTorch 实现
│   ├── modelDesign.py        # 基础编码/发射/接收模型
│   ├── final_solution.py     # 最终修复版（推荐）
│   ├── modelTrain.py         # 训练循环
│   └── config.py             # 系统配置
├── tensorflow/               # TensorFlow 实现
│   ├── modelDesign.py        # Transformer Encoder
│   └── modelTrain.py         # 端到端训练
└── README.md
```

## 📄 许可证 | License

MIT
