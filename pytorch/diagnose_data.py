import torch
import numpy as np

def diagnose_data():
    """诊断数据和任务设计"""
    print("=== 数据诊断 ===")
    
    # 加载数据
    H_train = np.load('./data_train/H_train.npy')
    print(f"信道数据形状: {H_train.shape}")
    print(f"信道数据类型: {H_train.dtype}")
    print(f"信道数据范围: {H_train.min():.4f} ~ {H_train.max():.4f}")
    
    # 检查数据分布
    print(f"信道数据均值: {H_train.mean():.4f}")
    print(f"信道数据标准差: {H_train.std():.4f}")
    
    # 生成随机比特
    batch_size = 16
    num_bits = 4608
    bits = torch.randint(0, 2, (batch_size, num_bits), dtype=torch.float32)
    
    print(f"\n比特数据形状: {bits.shape}")
    print(f"比特数据均值: {bits.mean():.4f} (应该接近0.5)")
    
    # 检查任务的可学习性
    print("\n=== 任务可学习性检查 ===")
    
    # 创建一个简单的基线：直接从输入预测输出
    class SimpleBaseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 直接从比特预测比特（应该能达到很高的准确率）
            self.net = torch.nn.Linear(num_bits, 1152)
            
        def forward(self, bits):
            return self.net(bits)
    
    # 测试基线模型
    baseline = SimpleBaseline()
    optimizer = torch.optim.Adam(baseline.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("测试基线模型（直接从输入比特预测输出比特）...")
    
    for i in range(20):
        # 使用相同的比特作为输入和目标（理想情况）
        output = baseline(bits)
        target = bits[:, :output.shape[1]]  # 截取对应长度
        
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = (torch.sigmoid(output) > 0.5).float()
            acc = (pred == target).float().mean()
        
        if i % 5 == 0:
            print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    print("\n=== 通信系统任务分析 ===")
    
    # 分析通信系统的信息理论限制
    print("分析：在无噪声情况下，通信系统应该能够完美重建信号")
    print("但在有噪声情况下，存在信息理论限制")
    
    # 检查SNR对性能的影响
    snr_values = [0, 5, 10, 15, 20]
    print(f"\n不同SNR下的理论性能限制：")
    
    for snr in snr_values:
        # 计算理论容量（简化）
        capacity = np.log2(1 + 10**(snr/10))
        print(f"SNR={snr}dB: 理论容量≈{capacity:.2f} bits/symbol")
    
    print("\n=== 建议的解决方案 ===")
    print("1. 检查数据预处理是否正确")
    print("2. 验证通信系统设计的合理性")
    print("3. 使用更简单的任务进行验证")
    print("4. 检查损失函数是否适合任务")
    print("5. 考虑使用预训练或迁移学习")

def test_simple_communication():
    """测试一个极简的通信系统"""
    print("\n=== 测试极简通信系统 ===")
    
    class MinimalCommunication(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 极简设计：直接编码-解码
            self.encoder = torch.nn.Linear(1152, 96)
            self.decoder = torch.nn.Linear(96, 1152)
            
        def forward(self, bits):
            # 编码
            encoded = self.encoder(bits)
            # 添加噪声
            noise = torch.randn_like(encoded) * 0.1
            noisy_encoded = encoded + noise
            # 解码
            decoded = self.decoder(noisy_encoded)
            return decoded
    
    model = MinimalCommunication()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("训练极简通信系统...")
    
    for i in range(50):
        # 生成随机比特
        bits = torch.randint(0, 2, (16, 1152), dtype=torch.float32)
        
        # 前向传播
        output = model(bits)
        loss = criterion(output, bits)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            pred = (torch.sigmoid(output) > 0.5).float()
            acc = (pred == bits).float().mean()
        
        if i % 10 == 0:
            print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
    
    print("极简系统测试完成")

if __name__ == "__main__":
    diagnose_data()
    test_simple_communication()