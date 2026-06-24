import torch
import torch.nn as nn
import numpy as np

# 创建一个极简版本来调试问题
class VerySimpleEncoder(nn.Module):
    def __init__(self, num_re=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18432 * 2, 512),  # 32*4*144*2 for complex
            nn.ReLU(),
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

class VerySimpleTransmitter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4608 + 96*2, 1024),  # bits + feedback
            nn.ReLU(),
            nn.Linear(1024, 32*144*2 + 5)  # output + control
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

class VerySimpleReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4*144*2 + 32*4*144*2, 1024),  # signal + channel
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
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

# 测试简化版本
def test_simple_model():
    print("测试简化模型...")
    
    # 创建模型
    encoder = VerySimpleEncoder()
    transmitter = VerySimpleTransmitter()
    receiver = VerySimpleReceiver()
    
    # 创建测试数据
    batch_size = 4
    H = torch.randn(batch_size, 32, 4, 144, dtype=torch.complex64)
    bits = torch.randint(0, 2, (batch_size, 4608), dtype=torch.float32)
    snr_dl = torch.randn(batch_size)
    snr_ul = torch.randn(batch_size)
    
    print("测试前向传播...")
    
    # 编码器
    U = encoder(H, snr_dl)
    print(f"编码器输出: {U.shape}")
    
    # 发射机
    X, ctrl = transmitter(bits, U, snr_dl)
    print(f"发射机输出: X={X.shape}, ctrl={ctrl.shape}")
    
    # 模拟信道
    Y = torch.randn(batch_size, 4, 1, 144, dtype=torch.complex64)
    
    # 接收机
    llr = receiver(Y, H, ctrl, snr_dl)
    print(f"接收机输出: {llr.shape}")
    
    # 测试训练
    print("\n测试训练过程...")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + 
        list(transmitter.parameters()) + 
        list(receiver.parameters()), 
        lr=0.01
    )
    criterion = nn.BCEWithLogitsLoss()
    
    for i in range(10):
        # 前向传播
        U = encoder(H, snr_dl)
        X, ctrl = transmitter(bits, U, snr_dl)
        llr = receiver(Y, H, ctrl, snr_dl)
        
        # 计算损失
        target = bits[:, :llr.shape[1]]
        loss = criterion(llr, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            pred = (torch.sigmoid(llr) > 0.5).float()
            acc = (pred == target).float().mean()
        
        print(f"Step {i+1}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    print("简化模型测试完成！")

if __name__ == "__main__":
    test_simple_model()