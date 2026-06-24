#=======================================================================================================================
#=======================================================================================================================
import torch
import torch.nn as nn
from modelDesign import *
import numpy as np
import pickle
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
# Link Defining
class link_train(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = Encoder()
        self._transmitter = Transmitter()
        self._receiver = Receiver()

        self._num_uplink_subcarriers = NUM_UPLINK_SUBCARRIERS
        self._num_uplink_symbols = NUM_UPLINK_SYMBOLS
        self._num_downlink_data_subcarriers = NUM_DOWNLINK_DATA_SUBCARRIERS
        self._num_downlink_data_symbols = NUM_DOWNLINK_DATA_SYMBOLS
        self._num_downlink_ctrl_bits = NUM_DOWNLINK_CTRL_BITS
        self._num_downlink_tx = NUM_DOWNLINK_TX

    def forward(self, h_train, b_data, snr_dl, snr_ul):
        batch_size = h_train.shape[0]
        H = torch.tensor(h_train)
        # ==============================================================================================================
        # Uplink========================================================================================================
        # ==============================================================================================================
        # Uplink transmitting
        U = self._encoder(H, snr_dl)
        # Dimension check
        assert U.shape == (batch_size, self._num_uplink_subcarriers * self._num_uplink_symbols), "Dimension error!"
        # Uplink norm
        energy = torch.mean(torch.abs(U) ** 2, dim=1, keepdim=True)
        U = U / torch.sqrt(energy)
        # Uplink channel
        g = torch.complex(torch.randn_like(U, dtype=torch.float32) / np.sqrt(2), torch.randn_like(U, dtype=torch.float32) / np.sqrt(2))
        n_ul = g * torch.sqrt(torch.reshape(10 ** (-snr_ul / 10.0), [-1, 1]))
        I = U + n_ul
        # ==============================================================================================================
        # ==============================================================================================================
        # ==============================================================================================================


        # ==============================================================================================================
        # Downlink======================================================================================================
        # ==============================================================================================================
        # Downlink transmitting
        X, b_ctrl = self._transmitter(b_data, I, snr_dl)
        # Dimension check
        assert X.shape == (batch_size, self._num_downlink_tx, self._num_downlink_data_symbols, self._num_downlink_data_subcarriers), "Dimension error!"
        assert b_ctrl.shape == (batch_size, self._num_downlink_ctrl_bits), "Dimension error!"
        if not torch.all(torch.eq(b_ctrl, b_ctrl * b_ctrl)):
            raise AssertionError("Ctrl bits format error!")
        # Downlink norm
        energy = torch.mean(torch.sum(torch.abs(X) ** 2, dim=1, keepdim=True), dim=(1, 2, 3), keepdim=True)
        X = X / torch.sqrt(energy)
        # Downlink channel
        X = X.unsqueeze(1)
        Y = torch.sum(H * X, dim=2)
        g = torch.complex(torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2), torch.randn_like(Y, dtype=torch.float32) / np.sqrt(2))
        n_dl = g * torch.sqrt(torch.reshape(10 ** (-snr_dl / 10.0), [-1, 1, 1, 1]))
        Y = Y + n_dl
        # Downlink receiving
        c_data = self._receiver(Y, H, b_ctrl, snr_dl)
        # ==============================================================================================================
        # ==============================================================================================================
        # ==============================================================================================================
        return c_data
#=======================================================================================================================
#=======================================================================================================================
# Data Loading
H_train = np.load('./data_train/H_train.npy')
print(H_train.shape)
#=======================================================================================================================
#=======================================================================================================================
# Training
BATCH_SIZE = 32
NUM_TRAINING_ITERATIONS = 2
model = link_train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for i in range(NUM_TRAINING_ITERATIONS):
    b_data = torch.randint(0, 2, (BATCH_SIZE, NUM_MAX_BITS), dtype=torch.float32)
    h_train = H_train[np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)]
    snr_dl = torch.randn(BATCH_SIZE) * (SNR_DL_RANGE[1] - SNR_DL_RANGE[0]) + SNR_DL_RANGE[0]
    snr_ul = torch.randn(BATCH_SIZE) * (SNR_UL_RANGE[1] - SNR_UL_RANGE[0]) + SNR_UL_RANGE[0]

    llr = model(h_train, b_data, snr_dl, snr_ul)
    loss = criterion(llr, b_data[:, :llr.shape[1]])

    optimizer.zero_grad()
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()

    acc = ((llr >= 0).float() == b_data[:, :llr.shape[1]].float()).sum() / llr.shape[0]/ llr.shape[1]
    print(f'Iteration {i}/{NUM_TRAINING_ITERATIONS}  Loss: {loss.item():.4f} Acc: {acc.item():.4f}')

torch.save(model._encoder.state_dict(), './modelSubmit/encoder.pth')
torch.save(model._transmitter.state_dict(), './modelSubmit/transmitter.pth')
torch.save(model._receiver.state_dict(), './modelSubmit/receiver.pth')