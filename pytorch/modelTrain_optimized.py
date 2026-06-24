import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.nn import functional as F

# 系统参数
NUM_UPLINK_SUBCARRIERS = 96
NUM_DOWNLINK_DATA_SUBCARRIERS = 144
NUM_DOWNLINK_TX = 32
NUM_DOWNLINK_RX = 4
NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS * 32
SNR_DL_RANGE = [-20, 20]
SNR_UL_RANGE = [-20, 0]

class SimpleEncoder(nn.Module):
    """简化编码器 - 专注于核心功能"""
    def __init__(self, num_re=96, input_dim=18432):  # 32*4*144
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),  # *2 for real/imag
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_re * 2)  # *2 for real/imag output
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


class SimpleTransmitter(nn.Module):
    """简化发射机"""
    def __init__(self, input_bits=4608, feedback_dim=96):  # 144*32
        super().__init__()
        self.data_net = nn.Sequential(
            nn.Linear(input_bits, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        self.feedback_net = nn.Sequential(
            nn.Linear(feedback_dim * 2, 256),  # *2 for complex
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 32 * 144 * 2)  # output size * 2 for complex
        )
        
        self.ctrl_net = nn.Sequential(
            nn.Linear(512 + 256, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
        
    def forward(self, bits, feedback, snr):
        batch_size = bits.shape[0]
        
        # 处理数据
        data_feat = self.data_net(bits)
        
        # 处理反馈
        fb_real = torch.real(feedback).reshape(batch_size, -1)
        fb_imag = torch.imag(feedback).reshape(batch_size, -1)
        fb_combined = torch.cat([fb_real, fb_imag], dim=-1)
        fb_feat = self.feedback_net(fb_combined)
        
        # 融合特征
        combined = torch.cat([data_feat, fb_feat], dim=-1)
        
        # 生成输出
        output = self.output_net(combined)
        real_part = output[:, :32*144]
        imag_part = output[:, 32*144:]
        
        x = torch.complex(real_part, imag_part)
        x = x.reshape(batch_size, 32, 1, 144)
        
        # 控制位
        ctrl = self.ctrl_net(combined)
        ctrl = (ctrl > 0.5).float()
        
        return x, ctrl


class SimpleReceiver(nn.Module):
    """简化接收机"""
    def __init__(self, output_bits=1152):  # 144*1*4*2
        super().__init__()
        signal_dim = 4 * 144 * 2  # 4 rx ant, 144 subcarriers, 2 for complex
        channel_dim = 32 * 4 * 144 * 2  # channel matrix
        
        self.signal_net = nn.Sequential(
            nn.Linear(signal_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.channel_net = nn.Sequential(
            nn.Linear(channel_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_bits)
        )
        
    def forward(self, y, h, ctrl, snr):
        batch_size = y.shape[0]
        
        # 处理接收信号
        y_real = torch.real(y).reshape(batch_size, -1)
        y_imag = torch.imag(y).reshape(batch_size, -1)
        y_combined = torch.cat([y_real, y_imag], dim=-1)
        signal_feat = self.signal_net(y_combined)
        
        # 处理信道
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        channel_feat = self.channel_net(h_combined)
        
        # 解码
        combined = torch.cat([signal_feat, channel_feat], dim=-1)
        llr = self.decoder(combined)
        
        return llr


class OptimizedLinkSystem(nn.Module):
    """优化的通信系统 - 使用简化架构"""
    def __init__(self):
        super().__init__()
        self._encoder = SimpleEncoder()
        self._transmitter = SimpleTransmitter()
        self._receiver = SimpleReceiver()

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


class WarmupCosineScheduler:
    """带预热的余弦退火调度器"""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 预热阶段
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            # 余弦退火阶段
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_optimized():
    """优化的训练函数"""
    # 创建目录
    os.makedirs('./modelSubmit', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 数据加载
    H_train = np.load('./data_train/H_train.npy')
    print(f"Training data shape: {H_train.shape}")
    
    # 模型初始化
    model = OptimizedLinkSystem()
    
    # 权重初始化
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # 训练参数 - 调整为更适合的设置
    BATCH_SIZE = 32
    NUM_ITERATIONS = 2000
    LEARNING_RATE = 5e-3  # 进一步提高学习率
    WEIGHT_DECAY = 1e-5  # 减少正则化
    WARMUP_STEPS = 200
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, NUM_ITERATIONS)
    
    # 损失函数 - 使用标签平滑
    class LabelSmoothingBCE(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            
        def forward(self, pred, target):
            # 标签平滑：0 -> smoothing/2, 1 -> 1-smoothing/2
            target_smooth = target * (1 - self.smoothing) + self.smoothing / 2
            return F.binary_cross_entropy_with_logits(pred, target_smooth)
    
    criterion = LabelSmoothingBCE(smoothing=0.1)
    
    # 训练记录
    train_losses = []
    train_accs = []
    best_acc = 0.0
    
    print("开始优化训练...")
    
    for i in tqdm(range(NUM_ITERATIONS), desc="Training"):
        model.train()
        
        # 数据采样
        b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
        h_train = H_train[np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)]
        
        # 课程学习 - 从高SNR开始，逐步降低
        progress = i / NUM_ITERATIONS
        if progress < 0.5:
            # 前50%使用高SNR，让模型先学会基本映射
            snr_dl = torch.rand(BATCH_SIZE) * 5 + 10  # 10-15 dB
            snr_ul = torch.rand(BATCH_SIZE) * 5 - 2.5 # -2.5-2.5 dB
        elif progress < 0.8:
            # 中间30%逐步降低SNR
            snr_dl = torch.rand(BATCH_SIZE) * 15 + 0  # 0-15 dB
            snr_ul = torch.rand(BATCH_SIZE) * 10 - 5  # -5-5 dB
        else:
            # 最后20%使用全范围SNR
            snr_dl = torch.rand(BATCH_SIZE) * 25 - 10 # -10-15 dB
            snr_ul = torch.rand(BATCH_SIZE) * 15 - 15 # -15-0 dB
        
        # 前向传播
        try:
            llr = model(h_train, b_data, snr_dl, snr_ul)
            target_bits = b_data[:, :llr.shape[1]]
            
            # 计算损失
            loss = criterion(llr, target_bits)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 - 放宽限制
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            lr = scheduler.step()
            
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
                print(f'  Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, LR: {lr:.6f}')
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
    plt.savefig('./logs/optimized_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n优化训练完成！最佳准确率: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 开始训练
    model = train_optimized()