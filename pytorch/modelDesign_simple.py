import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Linear


class SimpleEncoder(torch.nn.Module):
    """简化的编码器，更容易训练"""
    def __init__(self,
                 num_re=96,
                 num_tx_ant=32,
                 num_rx_ant=4,
                 num_data_subcarriers=144,
                 **kwargs):
        super(SimpleEncoder, self).__init__(**kwargs)

        self._num_re = num_re
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._num_data_subcarriers = num_data_subcarriers
        
        input_dim = self._num_tx_ant * self._num_rx_ant * self._num_data_subcarriers
        
        # 简化的网络结构
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # *2 for real and imag
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_re * 2)  # *2 for real and imag
        )
        
    def forward(self, h, snr):
        batch_size = h.shape[0]
        
        # 信道预处理
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        
        # 编码
        output = self.encoder_net(h_combined)
        
        # 分离实部和虚部
        output_real = output[:, :self._num_re]
        output_imag = output[:, self._num_re:]
        
        x = torch.complex(output_real, output_imag)
        return x


class SimpleTransmitter(torch.nn.Module):
    """简化的发射机"""
    def __init__(self, 
                 num_tx_ant=32, 
                 num_data_subcarriers=144, 
                 num_data_symbols=1,
                 **kwargs):
        super(SimpleTransmitter, self).__init__(**kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        
        input_bits_dim = 32 * num_data_subcarriers * num_data_symbols
        feedback_dim = 96
        output_dim = num_tx_ant * num_data_subcarriers * num_data_symbols
        
        # 数据处理网络
        self.data_net = nn.Sequential(
            nn.Linear(input_bits_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # 反馈处理网络
        self.feedback_net = nn.Sequential(
            nn.Linear(feedback_dim * 2, 128),  # *2 for complex
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 输出网络
        self.output_net = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim * 2)  # *2 for real and imag
        )
        
        # 控制位生成
        self.ctrl_net = nn.Sequential(
            nn.Linear(256 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.Sigmoid()
        )
        
    def forward(self, bits, feedback_info, snr):
        batch_size = bits.shape[0]
        
        # 数据处理
        data_features = self.data_net(bits)
        
        # 反馈处理
        feedback_real = torch.real(feedback_info).reshape(batch_size, -1)
        feedback_imag = torch.imag(feedback_info).reshape(batch_size, -1)
        feedback_combined = torch.cat([feedback_real, feedback_imag], dim=-1)
        feedback_features = self.feedback_net(feedback_combined)
        
        # 特征融合
        combined = torch.cat([data_features, feedback_features], dim=-1)
        
        # 输出生成
        output = self.output_net(combined)
        output_real = output[:, :output.shape[1]//2]
        output_imag = output[:, output.shape[1]//2:]
        
        x = torch.complex(output_real, output_imag)
        x = x.reshape(batch_size, self._num_tx_ant, self._num_data_symbols, self._num_data_subcarriers)
        
        # 控制位
        ctrl_bits = self.ctrl_net(combined)
        ctrl_bits = (ctrl_bits > 0.5).float()
        
        return x, ctrl_bits


class SimpleReceiver(torch.nn.Module):
    """简化的接收机"""
    def __init__(self, 
                 num_data_subcarriers=144, 
                 num_data_symbols=1, 
                 num_layers=4, 
                 num_bits_per_layer=2):
        super().__init__()
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._num_layers = num_layers
        self._num_bits_per_layer = num_bits_per_layer
        
        # 接收信号处理
        signal_dim = num_data_subcarriers * 4 * 2  # 4 rx antennas, 2 for complex
        channel_dim = 32 * 4 * num_data_subcarriers * 2  # channel info
        
        self.signal_net = nn.Sequential(
            nn.Linear(signal_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.channel_net = nn.Sequential(
            nn.Linear(channel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 解码网络
        self.decoder = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, num_data_symbols * num_data_subcarriers * num_layers * num_bits_per_layer)
        )
        
    def forward(self, y, h, ctrl_bits, snr):
        batch_size = y.shape[0]
        
        # 信号处理
        y_real = torch.real(y).reshape(batch_size, -1)
        y_imag = torch.imag(y).reshape(batch_size, -1)
        y_combined = torch.cat([y_real, y_imag], dim=-1)
        signal_features = self.signal_net(y_combined)
        
        # 信道处理
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        channel_features = self.channel_net(h_combined)
        
        # 特征融合和解码
        combined = torch.cat([signal_features, channel_features], dim=-1)
        llr = self.decoder(combined)
        
        return llr


# 为了兼容性，提供别名
Encoder = SimpleEncoder
Transmitter = SimpleTransmitter
Receiver = SimpleReceiver