import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 系统参数
NUM_UPLINK_SUBCARRIERS = 96
NUM_DOWNLINK_DATA_SUBCARRIERS = 144
NUM_DOWNLINK_TX = 32
NUM_DOWNLINK_RX = 4
NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS * 32
SNR_DL_RANGE = [-20, 20]
SNR_UL_RANGE = [-20, 0]

# 简化但有效的模型架构
class FixedEncoder(nn.Module):
    def __init__(self, num_re=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18432 * 2, 1024),  # 32*4*144*2 for complex
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_re * 2)
        )
        
    def forward(self, h, snr):
        batch_size = h.shape[0]
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        
        output = self.net(h_combined)
        real_part = output[:, :96]
        imag_part = output[:, 96:]
        return torch.complex(real_part, imag_part)

class FixedTransmitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4608 + 96*2, 1024),  # bits + feedback
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 32*144*2 + 5)  # output + control
        )
        
    def forward(self, bits, feedback, snr):
        batch_size = bits.shape[0]
        fb_real = torch.real(feedback).reshape(batch_size, -1)
        fb_imag = torch.imag(feedback).reshape(batch_size, -1)
        fb_combined = torch.cat([fb_real, fb_imag], dim=-1)
        
        combined = torch.cat([bits, fb_combined], dim=-1)
        output = self.net(combined)
        
        # 分离输出和控制位
        main_output = output[:, :-5]
        ctrl_bits = torch.sigmoid(output[:, -5:])
        ctrl_bits = (ctrl_bits > 0.5).float()
        
        # 重塑输出
        real_part = main_output[:, :32*144]
        imag_part = main_output[:, 32*144:]
        x = torch.complex(real_part, imag_part)
        x = x.reshape(batch_size, 32, 1, 144)
        
        return x, ctrl_bits

class FixedReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4*144*2 + 32*4*144*2, 1024),  # signal + channel
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1152)  # output bits
        )
        
    def forward(self, y, h, ctrl, snr):
        batch_size = y.shape[0]
        
        # 处理接收信号
        y_real = torch.real(y).reshape(batch_size, -1)
        y_imag = torch.imag(y).reshape(batch_size, -1)
        y_combined = torch.cat([y_real, y_imag], dim=-1)
        
        # 处理信道
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        
        # 合并输入
        combined = torch.cat([y_combined, h_combined], dim=-1)
        return self.net(combined)

class FixedLinkSystem(nn.Module):
    """修复后的通信系统"""
    def __init__(self):
        super().__init__()
        self._encoder = FixedEncoder()
        self._transmitter = FixedTransmitter()
        self._receiver = FixedReceiver()

    def forward(self, h_train, b_data, snr_dl, snr_ul):
        batch_size = h_train.shape[0]
        H = torch.tensor(h_train, dtype=torch.complex64)
        
        # Uplink - 编码器
        U = self._encoder(H, snr_dl)
        
        # 功率归一化
        energy = torch.mean(torch.abs(U) ** 2, dim=1, keepdim=True)
        U = U / torch.sqrt(energy + 1e-8)
        
        # 上行信道
        g = torch.complex(
            torch.randn_like(U, dtype=torch.float32) / np.sqrt(2), 
            torch.randn_like(U, dtype=torch.float32) / np.sqrt(2)
        )
        noise_power = 10 ** (-snr_ul / 10.0)
        n_ul = g * torch.sqrt(torch.reshape(noise_power, [-1, 1]))
        I = U + n_ul
        
        # Downlink - 发射机
        X, b_ctrl = self._transmitter(b_data, I, snr_dl)
        
        # 功率归一化
        energy = torch.mean(torch.sum(torch.abs(X) ** 2, dim=1, keepdim=True), dim=(1, 2, 3), keepdim=True)
        X = X / torch.sqrt(energy + 1e-8)
        
        # 下行信道
        X = X.unsqueeze(1)
        Y = torch.sum(H * X, dim=2)
        
        g = torch.complex(
            torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2), 
            torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2)
        )
        noise_power = 10 ** (-snr_dl / 10.0)
        n_dl = g * torch.sqrt(torch.reshape(noise_power, [-1, 1, 1, 1]))
        Y = Y + n_dl
        
        # 接收机
        c_data = self._receiver(Y, H, b_ctrl, snr_dl)
        return c_data

def train_fixed():
    """修复后的训练函数"""
    # 创建目录
    os.makedirs('./modelSubmit', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 数据加载
    H_train = np.load('./data_train/H_train.npy')
    print(f"Training data shape: {H_train.shape}")
    
    # 模型初始化
    model = FixedLinkSystem()
    
    # 训练参数
    BATCH_SIZE = 16
    NUM_ITERATIONS = 500  # 减少迭代次数，因为简化模型收敛更快
    LEARNING_RATE = 0.01  # 提高学习率
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练记录
    train_losses = []
    train_accs = []
    best_acc = 0.0
    
    print("开始修复后的训练...")
    
    for i in tqdm(range(NUM_ITERATIONS), desc="Training"):
        model.train()
        
        # 数据采样
        b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
        h_train = H_train[np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)]
        
        # 使用较好的SNR进行训练
        snr_dl = torch.rand(BATCH_SIZE) * 15 + 5   # 5-20 dB
        snr_ul = torch.rand(BATCH_SIZE) * 10 - 5   # -5-5 dB
        
        # 前向传播
        try:
            llr = model(h_train, b_data, snr_dl, snr_ul)
            target_bits = b_data[:, :llr.shape[1]]
            
            # 计算损失
            loss = criterion(llr, target_bits)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 计算准确率
            with torch.no_grad():
                predictions = (torch.sigmoid(llr) > 0.5).float()
                acc = (predictions == target_bits).float().mean()
            
            train_losses.append(loss.item())
            train_accs.append(acc.item())
            
            # 记录最佳模型
            if acc.item() > best_acc:
                best_acc = acc.item()
                torch.save({
                    'encoder': model._encoder.state_dict(),
                    'transmitter': model._transmitter.state_dict(),
                    'receiver': model._receiver.state_dict(),
                    'iteration': i,
                    'best_acc': best_acc
                }, './modelSubmit/best_model.pth')
            
            # 打印进度
            if (i + 1) % 50 == 0:
                avg_loss = np.mean(train_losses[-50:])
                avg_acc = np.mean(train_accs[-50:])
                print(f'\nIteration {i+1}/{NUM_ITERATIONS}')
                print(f'  Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                print(f'  Best Acc: {best_acc:.4f}')
                
        except Exception as e:
            print(f"Error at iteration {i}: {e}")
            continue
    
    # 保存最终模型
    torch.save(model._encoder.state_dict(), './modelSubmit/encoder.pth')
    torch.save(model._transmitter.state_dict(), './modelSubmit/transmitter.pth')
    torch.save(model._receiver.state_dict(), './modelSubmit/receiver.pth')
    
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
    plt.savefig('./logs/fixed_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n修复后训练完成！最佳准确率: {best_acc:.4f}")
    return model

if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 开始训练
    model = train_fixed()