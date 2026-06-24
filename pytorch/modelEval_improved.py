#=======================================================================================================================
#=======================================================================================================================
import torch
import torch.nn as nn
from modelDesign_improved import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
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
#=======================================================================================================================
#=======================================================================================================================

class ImprovedLinkSystem(nn.Module):
    """改进的端到端通信系统 - 与训练时保持一致"""
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


def evaluate_model_comprehensive():
    """全面的模型评估"""
    
    # 数据加载
    try:
        H_test = np.load('./data_train/H_train.npy')  # 使用训练数据进行测试
    except:
        H_test = np.load('../data_train/H_train.npy')  # 备用路径
    
    print(f"Test data shape: {H_test.shape}")
    
    # 模型加载
    model = ImprovedLinkSystem()
    
    try:
        # 尝试加载最佳模型
        checkpoint = torch.load('./modelSubmit/best_model.pth', map_location='cpu')
        model._encoder.load_state_dict(checkpoint['encoder'])
        model._transmitter.load_state_dict(checkpoint['transmitter'])
        model._receiver.load_state_dict(checkpoint['receiver'])
        print("Loaded best model checkpoint")
    except:
        # 回退到单独的模型文件
        model._encoder.load_state_dict(torch.load('./modelSubmit/encoder.pth', map_location='cpu'))
        model._transmitter.load_state_dict(torch.load('./modelSubmit/transmitter.pth', map_location='cpu'))
        model._receiver.load_state_dict(torch.load('./modelSubmit/receiver.pth', map_location='cpu'))
        print("Loaded individual model files")
    
    model.eval()
    
    # 评估参数
    NUM_TESTING_ITERATIONS = 1000
    BATCH_SIZE = 16
    
    # 不同SNR下的性能评估
    snr_dl_values = np.arange(-15, 16, 5)  # -15 to 15 dB, step 5
    snr_ul_values = np.arange(-15, 1, 5)   # -15 to 0 dB, step 5
    
    results = {
        'overall_score': 0.0,
        'snr_performance': {},
        'ber_vs_snr': {},
        'throughput_vs_snr': {}
    }
    
    print("开始全面评估...")
    
    # 总体性能评估
    total_score = 0.0
    total_iterations = 0
    
    with torch.no_grad():
        for i in tqdm(range(NUM_TESTING_ITERATIONS), desc="Overall Performance"):
            # 随机采样
            b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
            h_test = H_test[np.random.choice(H_test.shape[0], BATCH_SIZE, replace=False)]
            
            # 随机SNR
            snr_dl = torch.rand(BATCH_SIZE) * (SNR_DL_RANGE[1] - SNR_DL_RANGE[0]) + SNR_DL_RANGE[0]
            snr_ul = torch.rand(BATCH_SIZE) * (SNR_UL_RANGE[1] - SNR_UL_RANGE[0]) + SNR_UL_RANGE[0]
            
            # 前向传播
            llr = model(h_test, b_data, snr_dl, snr_ul)
            
            # 计算得分
            for j in range(BATCH_SIZE):
                B_max = b_data.shape[1]
                B = llr.shape[1]
                predictions = (torch.sigmoid(llr[j]) > 0.5).float()
                B_c = torch.sum((predictions == b_data[j, :B]).float())
                miu = 100.0 * (B_c + ((B_max - B) * 0.5)) / B_max
                total_score += miu.item()
                total_iterations += 1
    
    results['overall_score'] = total_score / total_iterations
    print(f"Overall Score: {results['overall_score']:.2f}")
    
    # 不同SNR下的详细性能评估
    print("\n评估不同SNR下的性能...")
    
    for snr_dl in tqdm(snr_dl_values, desc="SNR Analysis"):
        snr_key = f"SNR_DL_{snr_dl}"
        results['snr_performance'][snr_key] = []
        results['ber_vs_snr'][snr_key] = []
        results['throughput_vs_snr'][snr_key] = []
        
        with torch.no_grad():
            for _ in range(100):  # 每个SNR点测试100次
                b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
                h_test = H_test[np.random.choice(H_test.shape[0], BATCH_SIZE, replace=False)]
                
                # 固定下行SNR，随机上行SNR
                snr_dl_fixed = torch.full((BATCH_SIZE,), float(snr_dl))
                snr_ul_random = torch.rand(BATCH_SIZE) * (SNR_UL_RANGE[1] - SNR_UL_RANGE[0]) + SNR_UL_RANGE[0]
                
                llr = model(h_test, b_data, snr_dl_fixed, snr_ul_random)
                
                # 计算BER
                predictions = (torch.sigmoid(llr) > 0.5).float()
                target_bits = b_data[:, :llr.shape[1]]
                ber = 1 - (predictions == target_bits).float().mean()
                
                # 计算吞吐量 (成功传输的比特数)
                throughput = (predictions == target_bits).float().sum() / target_bits.numel()
                
                # 计算得分
                B_max = b_data.shape[1]
                B = llr.shape[1]
                B_c = torch.sum((predictions == target_bits).float())
                score = 100.0 * (B_c + ((B_max - B) * 0.5)) / B_max / BATCH_SIZE
                
                results['snr_performance'][snr_key].append(score.item())
                results['ber_vs_snr'][snr_key].append(ber.item())
                results['throughput_vs_snr'][snr_key].append(throughput.item())
    
    # 计算统计信息
    print("\n不同SNR下的性能统计:")
    print("SNR(dB) | Score(%) | BER      | Throughput")
    print("-" * 45)
    
    for snr_dl in snr_dl_values:
        snr_key = f"SNR_DL_{snr_dl}"
        avg_score = np.mean(results['snr_performance'][snr_key])
        avg_ber = np.mean(results['ber_vs_snr'][snr_key])
        avg_throughput = np.mean(results['throughput_vs_snr'][snr_key])
        
        print(f"{snr_dl:7d} | {avg_score:8.2f} | {avg_ber:.6f} | {avg_throughput:.6f}")
    
    # 绘制性能曲线
    plt.figure(figsize=(15, 5))
    
    # 得分 vs SNR
    plt.subplot(1, 3, 1)
    scores_mean = [np.mean(results['snr_performance'][f"SNR_DL_{snr}"]) for snr in snr_dl_values]
    scores_std = [np.std(results['snr_performance'][f"SNR_DL_{snr}"]) for snr in snr_dl_values]
    plt.errorbar(snr_dl_values, scores_mean, yerr=scores_std, marker='o', capsize=5)
    plt.xlabel('Downlink SNR (dB)')
    plt.ylabel('Score (%)')
    plt.title('Score vs SNR')
    plt.grid(True)
    
    # BER vs SNR
    plt.subplot(1, 3, 2)
    ber_mean = [np.mean(results['ber_vs_snr'][f"SNR_DL_{snr}"]) for snr in snr_dl_values]
    ber_std = [np.std(results['ber_vs_snr'][f"SNR_DL_{snr}"]) for snr in snr_dl_values]
    plt.errorbar(snr_dl_values, ber_mean, yerr=ber_std, marker='s', capsize=5, color='red')
    plt.xlabel('Downlink SNR (dB)')
    plt.ylabel('Bit Error Rate')
    plt.title('BER vs SNR')
    plt.yscale('log')
    plt.grid(True)
    
    # 吞吐量 vs SNR
    plt.subplot(1, 3, 3)
    throughput_mean = [np.mean(results['throughput_vs_snr'][f"SNR_DL_{snr}"]) for snr in snr_dl_values]
    throughput_std = [np.std(results['throughput_vs_snr'][f"SNR_DL_{snr}"]) for snr in snr_dl_values]
    plt.errorbar(snr_dl_values, throughput_mean, yerr=throughput_std, marker='^', capsize=5, color='green')
    plt.xlabel('Downlink SNR (dB)')
    plt.ylabel('Throughput')
    plt.title('Throughput vs SNR')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存结果
    os.makedirs('./evaluation_results', exist_ok=True)
    plt.savefig('./evaluation_results/performance_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存详细结果
    np.save('./evaluation_results/evaluation_results.npy', results)
    
    print(f"\n最终评估完成!")
    print(f"总体得分: {results['overall_score']:.2f}%")
    print(f"结果已保存到 ./evaluation_results/")
    
    return results


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 开始评估
    results = evaluate_model_comprehensive()