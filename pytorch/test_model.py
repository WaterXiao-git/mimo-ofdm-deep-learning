import torch
import numpy as np
from modelDesign_improved import *

# 测试模型初始化和前向传播
print('Testing model initialization...')
encoder = ImprovedEncoder()
transmitter = ImprovedTransmitter()
receiver = ImprovedReceiver()

# 创建测试数据
batch_size = 2
h = torch.randn(batch_size, 32, 4, 144, dtype=torch.complex64)
snr_dl = torch.randn(batch_size)
snr_ul = torch.randn(batch_size)
bits = torch.randint(0, 2, (batch_size, 4608), dtype=torch.float32)

print('Testing encoder...')
U = encoder(h, snr_dl)
print(f'Encoder output shape: {U.shape}')

print('Testing transmitter...')
X, ctrl = transmitter(bits, U, snr_dl)
print(f'Transmitter output shapes: X={X.shape}, ctrl={ctrl.shape}')

print('Testing receiver...')
Y = torch.randn(batch_size, 4, 1, 144, dtype=torch.complex64)
llr = receiver(Y, h, ctrl, snr_dl)
print(f'Receiver output shape: {llr.shape}')

print('All tests passed!')