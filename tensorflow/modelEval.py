#=======================================================================================================================
#=======================================================================================================================
from tensorflow import keras
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
NUM_MAX_BITS = NUM_DOWNLINK_DATA_SUBCARRIERS*NUM_DOWNLINK_DATA_SYMBOLS*32

SNR_DL_RANGE = [-20,20]
SNR_UL_RANGE = [-20,0]
#=======================================================================================================================
#=======================================================================================================================
# Link Defining
class link_train(tf.keras.Model):
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

    def call(self, h_train, b_data, snr_dl, snr_ul):
        batch_size = tf.shape(h_train)[0]
        H = h_train
        # ==============================================================================================================
        # Uplink========================================================================================================
        # ==============================================================================================================
        # Uplink transmitting
        U = self._encoder(H, snr_dl)
        # Dimension check
        tf.debugging.assert_equal(tf.shape(U), tf.constant([int(batch_size), self._num_uplink_subcarriers*self._num_uplink_symbols], dtype=tf.int32), message="Dimension error!")
        # Uplink norm
        energy = tf.reduce_mean(tf.square(tf.abs(U)), axis=(1), keepdims=True)
        U = U / tf.cast(tf.sqrt(energy), tf.complex64)
        # Uplink channel
        g = tf.complex(tf.random.normal(tf.shape(U), stddev=tf.sqrt(1 / 2)),tf.random.normal(tf.shape(U), stddev=tf.sqrt(1 / 2)))
        n_ul = g * tf.cast(tf.sqrt(tf.reshape(1 / tf.math.pow(10., snr_ul / 10.), [-1, 1])), dtype=tf.complex64)
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
        tf.debugging.assert_equal(tf.shape(X), tf.constant([int(batch_size), self._num_downlink_tx, self._num_downlink_data_symbols, self._num_downlink_data_subcarriers], dtype=tf.int32), message="Dimension error!")
        tf.debugging.assert_equal(tf.shape(b_ctrl), tf.constant([int(batch_size), self._num_downlink_ctrl_bits], dtype=tf.int32), message="Dimension error!")
        tf.debugging.assert_equal(b_ctrl*b_ctrl, b_ctrl, message="Ctrl bits format error!")
        # Downlink norm
        energy = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(X)), axis=(1), keepdims=True), axis=(1, 2, 3),keepdims=True)
        X = X / tf.cast(tf.sqrt(energy), tf.complex64)
        # Downlink channel
        X = tf.reshape(X, [tf.shape(X)[0],1,tf.shape(X)[1],tf.shape(X)[2],tf.shape(X)[3]])
        Y = tf.reduce_sum(H * X, axis=2)
        g = tf.complex(tf.random.normal(tf.shape(Y), stddev=tf.sqrt(1/2)), tf.random.normal(tf.shape(Y), stddev=tf.sqrt(1/2)))
        n_dl = g * tf.cast(tf.sqrt(tf.reshape(1 / tf.math.pow(10., snr_dl / 10.), [-1, 1, 1, 1])), dtype = tf.complex64)
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
H_train = np.load('../data_train/H_train.npy')
#=======================================================================================================================
#=======================================================================================================================
# Model Loading
model = link_train()
b_data = tf.random.uniform([2, NUM_MAX_BITS], 0, 2, tf.int32)
h_train = H_train[np.random.choice(H_train.shape[0], 2, replace=False)]
snr_dl = tf.random.uniform(shape=[1], minval=SNR_DL_RANGE[0], maxval=SNR_DL_RANGE[1])
snr_ul = tf.random.uniform(shape=[1], minval=SNR_UL_RANGE[0], maxval=SNR_UL_RANGE[1])
llr = model(h_train, b_data, snr_dl, snr_ul)

with open('modelSubmit/encoder', 'rb') as f:
    weights = pickle.load(f)
model._encoder.set_weights(weights)

with open('modelSubmit/transmitter', 'rb') as f:
    weights = pickle.load(f)
model._transmitter.set_weights(weights)

with open('modelSubmit/receiver', 'rb') as f:
    weights = pickle.load(f)
model._receiver.set_weights(weights)
#=======================================================================================================================
#=======================================================================================================================
# Testing
NUM_TESTING_ITERATIONS = 2
score = 0
for i in range(NUM_TESTING_ITERATIONS):
    print(i)
    b_data = tf.random.uniform([1, NUM_MAX_BITS], 0, 2, tf.int32)
    h_train = H_train[np.random.choice(H_train.shape[0], 1, replace=False)]
    snr_dl = tf.random.uniform(shape=[1], minval=SNR_DL_RANGE[0], maxval=SNR_DL_RANGE[1])
    snr_ul = tf.random.uniform(shape=[1], minval=SNR_UL_RANGE[0], maxval=SNR_UL_RANGE[1])
    llr = model(h_train, b_data, snr_dl, snr_ul)

    B_max = tf.shape(b_data)[1]
    B = tf.shape(llr)[1]
    B_c = (tf.keras.metrics.BinaryAccuracy(threshold=0)(b_data[:, :B], llr) * tf.cast(B,dtype=tf.float32)).numpy()
    miu = 100 * (B_c + tf.cast(B_max - B, dtype=tf.float32) * 0.5) / tf.cast(B_max,dtype=tf.float32)

    score = score + miu
score = np.asarray(score / NUM_TESTING_ITERATIONS)
print('score = ' + str(score))

