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

SNR_DL_RANGE = [-20, 20]
SNR_UL_RANGE = [-20, 0]
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
H_train = np.load('./data_train/H_train.npy')
print(H_train.shape)
#=======================================================================================================================
#=======================================================================================================================
# Training
BATCH_SIZE = 8
NUM_TRAINING_ITERATIONS = 2
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model = link_train()

for i in range(NUM_TRAINING_ITERATIONS):
    b_data = tf.random.uniform([BATCH_SIZE, NUM_MAX_BITS], 0, 2, tf.int32)
    h_train = H_train[np.random.choice(H_train.shape[0], BATCH_SIZE, replace=False)]
    snr_dl = tf.random.uniform(shape=[BATCH_SIZE], minval=SNR_DL_RANGE[0], maxval=SNR_DL_RANGE[1])
    snr_ul = tf.random.uniform(shape=[BATCH_SIZE], minval=SNR_UL_RANGE[0], maxval=SNR_UL_RANGE[1])
    with tf.GradientTape() as tape:
        llr = model(h_train, b_data, snr_dl, snr_ul)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(b_data[:, :tf.shape(llr)[1]], llr)
    weights = model.trainable_weights
    grads = tape.gradient(loss, weights)
    optimizer.apply_gradients(zip(grads, weights))

    acc = tf.keras.metrics.BinaryAccuracy(threshold=0)(b_data[:, :tf.shape(llr)[1]], llr)
    print('Iteration {}/{}  Loss: {:.4f} Acc: {:.4f}'.format(i, NUM_TRAINING_ITERATIONS, loss.numpy(), acc.numpy()))

weights = model._encoder.get_weights()
with open('./modelSubmit/encoder', 'wb') as f:
    pickle.dump(weights, f)

weights = model._transmitter.get_weights()
with open('./modelSubmit/transmitter', 'wb') as f:
    pickle.dump(weights, f)

weights = model._receiver.get_weights()
with open('./modelSubmit/receiver', 'wb') as f:
    pickle.dump(weights, f)

