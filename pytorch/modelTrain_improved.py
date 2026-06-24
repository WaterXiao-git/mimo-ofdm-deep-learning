#=======================================================================================================================
#=======================================================================================================================
import torch
import torch.nn as nn
from modelDesign_improved import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import os
#=======================================================================================================================
#=======================================================================================================================
# System Parameters Setting
NUM_UPLINK_SUBCARRIERS = 96
NUM_UPLINK_SYMBOLS = 1

NUM_DOWNLINK_DATA_SUBCARRIERS = 144
NUM_DOWNLINK_DATA_SYMBOLS = 1
NUM_DOWNLINK_CTRL_BITS = 5
NUM_DOWNLINK_TX = 32
NUM_DOWNLINK_RX = 4

SEPARATE_REAL_IMAG = 2
NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS * NUM_DOWNLINK_DATA_SYMBOLS * 32

SNR_DL_RANGE = [-20, 20]
SNR_UL_RANGE = [-20, 0]

# 训练参数
BATCH_SIZE = 16  # 减小批量大小以提高稳定性
NUM_TRAINING_ITERATIONS = 2000  # 增加训练轮数
LEARNING_RATE = 5e-4  # 降低学习率
WEIGHT_DECAY = 1e-6  # 减小权重衰减
PATIENCE = 100  # 增加早停耐心值
#=======================================================================================================================
#=======================================================================================================================

class ImprovedLinkSystem(nn.Module):
    """改进的端到端通信系统"""
    def __init__(self):
        super().__init__()
        self._encoder = ImprovedEncoder()
        self._transmitter = ImprovedTransmitter()
        self._receiver = ImprovedReceiver()

        self._num_uplink_subcarriers = NUM_UPLINK_SUBCARRIERS
        self._num_uplink_symbols = NUM_UPLINK_SYMBOLS
        self._num_downlink_data_subcarriers = NUM_DOWNLINK_DATA_SUBCARRIERS
        self._num_downlink_data_symbols = NUM_DOWNLINK_DATA_SYMBOLS
        self._num_downlink_ctrl_bits = NUM_DOWNLINK_CTRL_BITS
        self._num_downlink_tx = NUM_DOWNLINK_TX

    def forward(self, h_train, b_data, snr_dl, snr_ul):
        batch_size = h_train.shape[0]
        H = torch.tensor(h_train, dtype=torch.complex64)
        
        # ==============================================================================================================
        # Uplink========================================================================================================
        # ==============================================================================================================
        # Uplink transmitting
        U = self._encoder(H, snr_dl)
        # Dimension check
        assert U.shape == (batch_size, self._num_uplink_subcarriers * self._num_uplink_symbols), "Dimension error!"
        
        # Uplink norm with learnable power control
        energy = torch.mean(torch.abs(U) ** 2, dim=1, keepdim=True)
        U = U / torch.sqrt(energy + 1e-8)  # 添加小常数避免除零
        
        # Uplink channel with improved noise modeling
        g = torch.complex(
            torch.randn_like(U, dtype=torch.float32) / np.sqrt(2), 
            torch.randn_like(U, dtype=torch.float32) / np.sqrt(2)
        )
        noise_power = 10 ** (-snr_ul / 10.0)
        n_ul = g * torch.sqrt(torch.reshape(noise_power, [-1, 1]))
        I = U + n_ul
        # ==============================================================================================================

        # ==============================================================================================================
        # Downlink======================================================================================================
        # ==============================================================================================================
        # Downlink transmitting
        X, b_ctrl = self._transmitter(b_data, I, snr_dl)
        # Dimension check
        assert X.shape == (batch_size, self._num_downlink_tx, self._num_downlink_data_symbols, self._num_downlink_data_subcarriers), "Dimension error!"
        assert b_ctrl.shape == (batch_size, self._num_downlink_ctrl_bits), "Dimension error!"
        
        # Downlink norm with adaptive power allocation
        energy = torch.mean(torch.sum(torch.abs(X) ** 2, dim=1, keepdim=True), dim=(1, 2, 3), keepdim=True)
        X = X / torch.sqrt(energy + 1e-8)
        
        # Downlink channel
        X = X.unsqueeze(1)
        Y = torch.sum(H * X, dim=2)
        
        # Improved noise modeling
        g = torch.complex(
            torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2), 
            torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2)
        )
        noise_power = 10 ** (-snr_dl / 10.0)
        n_dl = g * torch.sqrt(torch.reshape(noise_power, [-1, 1, 1, 1]))
        Y = Y + n_dl
        
        # Downlink receiving
        c_data = self._receiver(Y, H, b_ctrl, snr_dl)
        # ==============================================================================================================
        
        return c_data


class FocalLoss(nn.Module):
    """Focal Loss用于处理不平衡数据"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def train_model():
    """改进的训练函数"""
    # 创建保存目录
    os.makedirs('./modelSubmit', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 数据加载
    H_train = np.load('./data_train/H_train.npy')
    print(f"Training data shape: {H_train.shape}")
    
    # 模型初始化
    model = ImprovedLinkSystem()
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_TRAINING_ITERATIONS)
    
    # 损失函数 - 使用更简单的BCE损失
    criterion = nn.BCEWithLogitsLoss()
    
    # 早停
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # 训练记录
    train_losses = []
    train_accs = []
    best_acc = 0.0
    
    print("开始训练...")
    
    for i in range(NUM_TRAINING_ITERATIONS):
        model.train()
        
        # 数据采样
        b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
        h_train = H_train[np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)]
        
        # SNR采样 - 使用更真实的分布
        snr_dl = torch.randn(BATCH_SIZE) * 5 + 5  # 均值5dB，标准差5dB
        snr_ul = torch.randn(BATCH_SIZE) * 3 - 10  # 均值-10dB，标准差3dB
        
        # 限制SNR范围
        snr_dl = torch.clamp(snr_dl, SNR_DL_RANGE[0], SNR_DL_RANGE[1])
        snr_ul = torch.clamp(snr_ul, SNR_UL_RANGE[0], SNR_UL_RANGE[1])

        # 前向传播
        llr = model(h_train, b_data, snr_dl, snr_ul)
        
        # 计算损失
        target_bits = b_data[:, :llr.shape[1]]
        loss = criterion(llr, target_bits)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 计算准确率
        with torch.no_grad():
            predictions = (torch.sigmoid(llr) > 0.5).float()
            acc = (predictions == target_bits).float().mean()
        
        train_losses.append(loss.item())
        train_accs.append(acc.item())
        
        # 打印进度
        if (i + 1) % 50 == 0:
            avg_loss = np.mean(train_losses[-50:])
            avg_acc = np.mean(train_accs[-50:])
            lr = optimizer.param_groups[0]['lr']
            
            print(f'Iteration {i+1}/{NUM_TRAINING_ITERATIONS}')
            print(f'  Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, LR: {lr:.6f}')
            
            # 保存最佳模型
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save({
                    'encoder': model._encoder.state_dict(),
                    'transmitter': model._transmitter.state_dict(),
                    'receiver': model._receiver.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': i,
                    'best_acc': best_acc
                }, './modelSubmit/best_model.pth')
                print(f'  New best accuracy: {best_acc:.4f} - Model saved!')
        
        # 早停检查
        if early_stopping(loss.item()):
            print(f"Early stopping at iteration {i+1}")
            break
    
    # 保存最终模型
    torch.save(model._encoder.state_dict(), './modelSubmit/encoder.pth')
    torch.save(model._transmitter.state_dict(), './modelSubmit/transmitter.pth')
    torch.save(model._receiver.state_dict(), './modelSubmit/receiver.pth')
    
    # 保存训练历史
    np.save('./logs/train_losses.npy', np.array(train_losses))
    np.save('./logs/train_accs.npy', np.array(train_accs))
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs)
    plt.title('Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('./logs/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"训练完成！最佳准确率: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 开始训练
    model = train_model()