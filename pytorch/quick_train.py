import torch
import torch.nn as nn
from modelDesign_improved import *
import numpy as np
import os

# 系统参数
NUM_UPLINK_SUBCARRIERS = 96
NUM_DOWNLINK_DATA_SUBCARRIERS = 144
NUM_DOWNLINK_TX = 32
NUM_DOWNLINK_RX = 4
NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS * 32
SNR_DL_RANGE = [-20, 20]
SNR_UL_RANGE = [-20, 0]

class SimpleLinkSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = ImprovedEncoder()
        self._transmitter = ImprovedTransmitter()
        self._receiver = ImprovedReceiver()

    def forward(self, h_train, b_data, snr_dl, snr_ul):
        batch_size = h_train.shape[0]
        H = torch.tensor(h_train, dtype=torch.complex64)
        
        # Uplink
        U = self._encoder(H, snr_dl)
        energy = torch.mean(torch.abs(U) ** 2, dim=1, keepdim=True)
        U = U / torch.sqrt(energy + 1e-8)
        
        g = torch.complex(
            torch.randn_like(U, dtype=torch.float32) / np.sqrt(2), 
            torch.randn_like(U, dtype=torch.float32) / np.sqrt(2)
        )
        noise_power = 10 ** (-snr_ul / 10.0)
        n_ul = g * torch.sqrt(torch.reshape(noise_power, [-1, 1]))
        I = U + n_ul
        
        # Downlink
        X, b_ctrl = self._transmitter(b_data, I, snr_dl)
        energy = torch.mean(torch.sum(torch.abs(X) ** 2, dim=1, keepdim=True), dim=(1, 2, 3), keepdim=True)
        X = X / torch.sqrt(energy + 1e-8)
        
        X = X.unsqueeze(1)
        Y = torch.sum(H * X, dim=2)
        
        g = torch.complex(
            torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2), 
            torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2)
        )
        noise_power = 10 ** (-snr_dl / 10.0)
        n_dl = g * torch.sqrt(torch.reshape(noise_power, [-1, 1, 1, 1]))
        Y = Y + n_dl
        
        c_data = self._receiver(Y, H, b_ctrl, snr_dl)
        return c_data

def quick_train():
    # 创建目录
    os.makedirs('./modelSubmit', exist_ok=True)
    
    # 数据加载
    H_train = np.load('./data_train/H_train.npy')
    print(f"Training data shape: {H_train.shape}")
    
    # 模型和优化器
    model = SimpleLinkSystem()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练参数
    BATCH_SIZE = 8
    NUM_ITERATIONS = 200
    
    print("开始快速训练验证...")
    
    best_acc = 0.0
    for i in range(NUM_ITERATIONS):
        model.train()
        
        # 数据采样
        b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
        h_train = H_train[np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)]
        
        # 使用固定的SNR范围进行训练
        snr_dl = torch.rand(BATCH_SIZE) * 10 - 5  # -5 to 5 dB
        snr_ul = torch.rand(BATCH_SIZE) * 10 - 15  # -15 to -5 dB
        
        # 前向传播
        llr = model(h_train, b_data, snr_dl, snr_ul)
        target_bits = b_data[:, :llr.shape[1]]
        loss = criterion(llr, target_bits)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            predictions = (torch.sigmoid(llr) > 0.5).float()
            acc = (predictions == target_bits).float().mean()
        
        if (i + 1) % 20 == 0:
            print(f'Iteration {i+1}/{NUM_ITERATIONS}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}')
            
            if acc.item() > best_acc:
                best_acc = acc.item()
                # 保存模型
                torch.save(model._encoder.state_dict(), './modelSubmit/encoder.pth')
                torch.save(model._transmitter.state_dict(), './modelSubmit/transmitter.pth')
                torch.save(model._receiver.state_dict(), './modelSubmit/receiver.pth')
                print(f'  New best accuracy: {best_acc:.4f} - Model saved!')
    
    print(f"快速训练完成！最佳准确率: {best_acc:.4f}")
    return model

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    model = quick_train()