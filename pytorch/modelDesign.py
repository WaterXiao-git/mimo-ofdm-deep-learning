import numpy as np
import torch
from einops import rearrange, repeat
from torch.nn import functional as F
from torch.nn import Linear


class Encoder(torch.nn.Module):
    def __init__(self,
                 num_re=96,
                 num_tx_ant=32,
                 num_rx_ant=4,
                 num_data_subcarriers=144,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self._num_re = num_re
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._num_data_subcarriers = num_data_subcarriers

        self._dense_real = Linear(self._num_tx_ant*self._num_rx_ant*self._num_data_subcarriers, num_re)
        self._dense_imag = Linear(self._num_tx_ant*self._num_rx_ant*self._num_data_subcarriers, num_re)
    def forward(self, h, snr):
        # Controlling===================================================================================================
        # Scheme switching according to h and snr

        # Scheme 1======================================================================================================
        h_real = torch.real(h)
        h_imag = torch.imag(h)
        x_real = self._dense_real(h_real.reshape(h.shape[0], -1))
        x_imag = self._dense_imag(h_imag.reshape(h.shape[0], -1))
        x = torch.complex(x_real, x_imag)
        # Scheme 2======================================================================================================
        # ...

        # Scheme 3======================================================================================================
        # ...

        # ...
        return x


class Transmitter(torch.nn.Module):
    def __init__(self, num_tx_ant=32, num_data_subcarriers=144, num_data_symbols=1, **kwargs):
        super(Transmitter, self).__init__(**kwargs)
        self._num_tx_ant = num_tx_ant
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._dense = Linear(32 * num_data_subcarriers * num_data_symbols, num_tx_ant * num_data_subcarriers * num_data_symbols)

    def forward(self, bits, feedback_info, snr):
        # Controlling===================================================================================================
        # Scheme switching according to feedback_info and snr

        # Scheme 1======================================================================================================
        # x = torch.ones(bits.shape[0], self._num_tx_ant, self._num_data_symbols, self._num_data_subcarriers)
        x = self._dense(bits)
        x = x.reshape(bits.shape[0], self._num_tx_ant, self._num_data_symbols, self._num_data_subcarriers)
        ctrl_bits = torch.ones((x.shape[0], 5))
        # Scheme 2======================================================================================================
        # ...

        # Scheme 3======================================================================================================
        # ...

        # ...
        return x, ctrl_bits


class Receiver(torch.nn.Module):
    def __init__(self, num_data_subcarriers=144, num_data_symbols=1, num_layers=4, num_bits_per_layer=2):
        super().__init__()
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._num_layers = num_layers
        self._num_bits_per_layer = num_bits_per_layer

    def forward(self, y, h, ctrl_bits, snr):
        # Controlling===================================================================================================
        # Scheme switching according to feedback_info and snr

        # Scheme 1======================================================================================================
        llr = torch.ones(y.shape[0], self._num_data_symbols * self._num_data_subcarriers * self._num_layers * self._num_bits_per_layer)
        # Scheme 2======================================================================================================
        # ...

        # Scheme 3======================================================================================================
        # ...

        # ...
        return llr
