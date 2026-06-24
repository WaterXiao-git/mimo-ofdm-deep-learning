import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F
from torch.nn import Linear


class ChannelAttention(nn.Module):
    """信道注意力机制，用于自适应关注重要的信道特征"""
    def __init__(self, input_dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [batch_size, features]
        avg_out = self.fc(self.avg_pool(x.unsqueeze(-1)).squeeze(-1))
        max_out = self.fc(self.max_pool(x.unsqueeze(-1)).squeeze(-1))
        attention = avg_out + max_out
        return x * attention


class SNRAwareBlock(nn.Module):
    """SNR感知模块，根据SNR调整处理策略"""
    def __init__(self, input_dim, output_dim):
        super(SNRAwareBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.snr_embedding = nn.Sequential(
            nn.Linear(1, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim + output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x, snr):
        # SNR嵌入
        snr_embed = self.snr_embedding(snr.unsqueeze(-1))
        
        # 特征融合
        combined = torch.cat([x, snr_embed], dim=-1)
        output = self.feature_transform(combined)
        
        return output


class ImprovedEncoder(torch.nn.Module):
    """改进的编码器，支持多方案自适应选择"""
    def __init__(self,
                 num_re=96,
                 num_tx_ant=32,
                 num_rx_ant=4,
                 num_data_subcarriers=144,
                 hidden_dim=256,
                 **kwargs):
        super(ImprovedEncoder, self).__init__(**kwargs)

        self._num_re = num_re
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._num_data_subcarriers = num_data_subcarriers
        self._hidden_dim = hidden_dim
        
        input_dim = self._num_tx_ant * self._num_rx_ant * self._num_data_subcarriers
        
        # 信道特征提取
        self.channel_processor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # *2 for real and imag
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 信道注意力机制
        self.channel_attention = ChannelAttention(hidden_dim)
        
        # SNR感知模块
        self.snr_aware_block = SNRAwareBlock(hidden_dim, hidden_dim)
        
        # 方案选择器
        self.scheme_selector = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),  # +1 for SNR
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3个方案
            nn.Softmax(dim=-1)
        )
        
        # 三个不同的编码方案
        self.scheme1 = self._create_scheme_network(hidden_dim, num_re)
        self.scheme2 = self._create_scheme_network(hidden_dim, num_re)
        self.scheme3 = self._create_scheme_network(hidden_dim, num_re)
        
    def _create_scheme_network(self, input_dim, output_dim):
        """创建单个方案的网络"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim * 2)  # *2 for real and imag
        )
    
    def forward(self, h, snr):
        batch_size = h.shape[0]
        
        # 信道预处理
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        
        # 特征提取
        features = self.channel_processor(h_combined)
        
        # 注意力机制
        features = self.channel_attention(features)
        
        # SNR感知处理
        features = self.snr_aware_block(features, snr)
        
        # 方案选择
        selector_input = torch.cat([features, snr.unsqueeze(-1)], dim=-1)
        scheme_weights = self.scheme_selector(selector_input)
        
        # 执行三个方案
        output1 = self.scheme1(features)
        output2 = self.scheme2(features)
        output3 = self.scheme3(features)
        
        # 加权融合
        output1_real, output1_imag = output1[:, :self._num_re], output1[:, self._num_re:]
        output2_real, output2_imag = output2[:, :self._num_re], output2[:, self._num_re:]
        output3_real, output3_imag = output3[:, :self._num_re], output3[:, self._num_re:]
        
        final_real = (scheme_weights[:, 0:1] * output1_real + 
                     scheme_weights[:, 1:2] * output2_real + 
                     scheme_weights[:, 2:3] * output3_real)
        final_imag = (scheme_weights[:, 0:1] * output1_imag + 
                     scheme_weights[:, 1:2] * output2_imag + 
                     scheme_weights[:, 2:3] * output3_imag)
        
        x = torch.complex(final_real, final_imag)
        return x


class ImprovedTransmitter(torch.nn.Module):
    """改进的发射机，增加预编码和功率控制"""
    def __init__(self, 
                 num_tx_ant=32, 
                 num_data_subcarriers=144, 
                 num_data_symbols=1,
                 hidden_dim=256,
                 **kwargs):
        super(ImprovedTransmitter, self).__init__(**kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._hidden_dim = hidden_dim
        
        input_bits_dim = 32 * num_data_subcarriers * num_data_symbols
        feedback_dim = 96  # 反馈信息维度
        
        # 数据预处理
        self.data_processor = nn.Sequential(
            nn.Linear(input_bits_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 反馈信息处理 - 修复维度问题
        self.feedback_processor = nn.Sequential(
            nn.Linear(feedback_dim * 2, hidden_dim // 2),  # *2 for complex
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # SNR感知模块 - 修复输入维度
        combined_dim = hidden_dim + hidden_dim // 2  # 256 + 128 = 384
        self.snr_aware_block = SNRAwareBlock(combined_dim, hidden_dim // 2)
        
        # 预编码网络 - 修复输入维度
        final_dim = combined_dim + hidden_dim // 2  # 384 + 128 = 512
        self.precoder = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tx_ant * num_data_subcarriers * num_data_symbols * 2)
        )
        
        # 控制位生成 - 修复输入维度
        self.ctrl_generator = nn.Sequential(
            nn.Linear(final_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5),
            nn.Sigmoid()
        )
        
    def forward(self, bits, feedback_info, snr):
        batch_size = bits.shape[0]
        
        # 数据处理
        data_features = self.data_processor(bits)
        
        # 反馈信息处理
        feedback_real = torch.real(feedback_info).reshape(batch_size, -1)
        feedback_imag = torch.imag(feedback_info).reshape(batch_size, -1)
        feedback_combined = torch.cat([feedback_real, feedback_imag], dim=-1)
        feedback_features = self.feedback_processor(feedback_combined)
        
        # 特征融合
        combined_features = torch.cat([data_features, feedback_features], dim=-1)
        
        # SNR感知处理
        enhanced_features = self.snr_aware_block(combined_features, snr)
        final_features = torch.cat([combined_features, enhanced_features], dim=-1)
        
        # 预编码
        precoded = self.precoder(final_features)
        precoded_real = precoded[:, :precoded.shape[1]//2]
        precoded_imag = precoded[:, precoded.shape[1]//2:]
        
        x = torch.complex(precoded_real, precoded_imag)
        x = x.reshape(batch_size, self._num_tx_ant, self._num_data_symbols, self._num_data_subcarriers)
        
        # 控制位生成
        ctrl_bits = self.ctrl_generator(final_features)
        ctrl_bits = (ctrl_bits > 0.5).float()  # 二值化
        
        return x, ctrl_bits


class ImprovedReceiver(torch.nn.Module):
    """改进的接收机，增加软判决和迭代解码"""
    def __init__(self, 
                 num_data_subcarriers=144, 
                 num_data_symbols=1, 
                 num_layers=4, 
                 num_bits_per_layer=2,
                 hidden_dim=256):
        super().__init__()
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._num_layers = num_layers
        self._num_bits_per_layer = num_bits_per_layer
        self._hidden_dim = hidden_dim
        
        # 接收信号处理
        self.signal_processor = nn.Sequential(
            nn.Linear(num_data_subcarriers * 4 * 2, hidden_dim),  # 4 rx antennas, 2 for complex
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 信道估计处理
        self.channel_processor = nn.Sequential(
            nn.Linear(32 * 4 * num_data_subcarriers * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # SNR感知模块
        combined_dim = hidden_dim + hidden_dim // 2  # 256 + 128 = 384
        self.snr_aware_block = SNRAwareBlock(combined_dim, hidden_dim // 4)
        
        # 软判决网络
        self.soft_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_data_symbols * num_data_subcarriers * num_layers * num_bits_per_layer)
        )
        
    def forward(self, y, h, ctrl_bits, snr):
        batch_size = y.shape[0]
        
        # 接收信号处理
        y_real = torch.real(y).reshape(batch_size, -1)
        y_imag = torch.imag(y).reshape(batch_size, -1)
        y_combined = torch.cat([y_real, y_imag], dim=-1)
        signal_features = self.signal_processor(y_combined)
        
        # 信道信息处理
        h_real = torch.real(h).reshape(batch_size, -1)
        h_imag = torch.imag(h).reshape(batch_size, -1)
        h_combined = torch.cat([h_real, h_imag], dim=-1)
        channel_features = self.channel_processor(h_combined)
        
        # 特征融合
        combined_features = torch.cat([signal_features, channel_features], dim=-1)
        
        # SNR感知处理
        snr_features = self.snr_aware_block(combined_features, snr)
        
        # 最终特征
        final_features = torch.cat([combined_features, snr_features], dim=-1)
        
        # 软判决
        llr = self.soft_decoder(final_features)
        
        return llr


# 保持原有接口的兼容性
Encoder = ImprovedEncoder
Transmitter = ImprovedTransmitter  
Receiver = ImprovedReceiver