import numpy as np
import tensorflow as tf
from tensorflow import keras
from einops import rearrange, repeat
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Add, MultiHeadAttention
from tensorflow.nn import relu
from tensorflow.keras import backend as K
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def get_angles(pos, i, embedding_dim):
    get_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
    return pos * get_rates

def positional_encoding(sequence_length, embedding_dim):
    angle_rads = get_angles(np.arange(sequence_length)[:, np.newaxis],
                            np.arange(embedding_dim)[np.newaxis, :],
                            embedding_dim)
    sines = np.sin(angle_rads[:, 0::2])
    coses = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, coses], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    return pos_encoding

class linear_embedding(Layer):
    def __init__(self, embedding_dim):
        super(linear_embedding, self).__init__()
        self._dense = Dense(embedding_dim)
        self._embedding_dim = embedding_dim
    def call(self, x):
        x = self._dense(x)
        return x * tf.math.sqrt(tf.cast(self._embedding_dim, dtype=tf.float32))

class FeedForward(Layer):
    def __init__(self, inner_dim, embedding_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()
        self._layer_norm = LayerNormalization()
        self._dense1 = Dense(inner_dim, activation='relu')
        self._dropout1 = Dropout(dropout_rate)
        self._dense2 = Dense(embedding_dim)
        self._dropout2 = Dropout(dropout_rate)
        self._add = Add()

    def call(self, x):
        short_cut = x
        x = self._dropout1(self._dense1(self._layer_norm(x)))
        x = self._dropout2(self._dense2(x))
        x = self._add([short_cut, x])
        return x

class AttentionBlock(Layer):
    def __init__(self, num_heads, embedding_dim, dropout_rate=0.0):
        super(AttentionBlock, self).__init__()
        self._layer_norm = LayerNormalization()
        self._mha = MultiHeadAttention(num_heads, embedding_dim, dropout=dropout_rate)
        self._dropout = Dropout(dropout_rate)
        self._add = Add()

    def call(self, x):
        short_cut = x
        x = self._layer_norm(x)
        x = self._mha(x, x)
        x = self._dropout(x)
        x = self._add([short_cut, x])
        return x

class TfmBlock(Layer):
  def __init__(self, num_head, embedding_dim, dropout_rate=0.0):
    super(TfmBlock, self).__init__()
    self._attblock = AttentionBlock(num_head, embedding_dim, dropout_rate)
    self._ffn = FeedForward(embedding_dim*2, embedding_dim, dropout_rate)

  @tf.function
  def call(self, x):
    x = self._attblock(x)
    x = self._ffn(x)
    return x

class ResBlock(Layer):
    def __init__(self, channel=128):
        super().__init__()
        self._channel = channel
    def build(self, input_shape):
        self._conv_1 = layers.Conv2D(self._channel, (3, 3), padding='SAME', activation=None)
        self._norm_1 = layers.LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = layers.Conv2D(self._channel, (3, 3), padding='SAME', activation=None)
        self._norm_2 = layers.LayerNormalization(axis=(-1, -2, -3))
    def call(self, inputs):
        x_ini = inputs
        x = self._norm_1(x_ini)
        x = relu(x)
        x = self._conv_1(x)
        x = self._norm_2(x)
        x = relu(x)
        x = self._conv_2(x)
        x_ini = layers.Add()([x_ini, x])
        return x_ini

class Encoder(Layer):
    def __init__(self,
                 num_re=96,
                 num_tx_ant=32,
                 embedding_dim=128,
                 num_head=8,
                 num_block=5,
                 dropout_rate=0.0,
                 num_sc_per_sb = 48,
                 num_layers = 4,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._num_sc_per_sb = num_sc_per_sb
        self._num_layers = num_layers

        self._num_block = num_block
        self._embedding = linear_embedding(embedding_dim)
        self._pos_encoding = positional_encoding(1000, embedding_dim)
        self._dropout = Dropout(dropout_rate)
        self._enc_layers = [TfmBlock(num_head, embedding_dim, dropout_rate)
                            for _ in range(num_block)]
        self._layer_norm = LayerNormalization()
        self._dense1 = Dense(num_tx_ant * 2)
        self._dense2 = Dense(units=num_re * 2, activation='sigmoid')

    def call(self, h, snr):
        # Controlling===================================================================================================
        # Scheme switching according to h and snr

        # Scheme 1======================================================================================================
        # From full channels to eigenvectors
        h = tf.reshape(h, [tf.shape(h)[0], 1, tf.shape(h)[1], 1, tf.shape(h)[2], tf.shape(h)[3], tf.shape(h)[4]])
        h_pc = tf.transpose(h, [3, 1, 2, 4, 5, 6, 0])
        h_pc_desired = h_pc
        h_pc_desired = tf.squeeze(h_pc_desired, axis=1)
        h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])
        h_pc_desired = tf.cast(h_pc_desired, dtype=tf.complex64)
        r = tf.matmul(h_pc_desired, h_pc_desired, adjoint_a=True)
        r = rearrange(r, 'batch nt ofdm (nsb nsc) tx1 tx2 -> batch nt ofdm nsb nsc tx1 tx2', nsc=self._num_sc_per_sb)
        r_avg = tf.reduce_mean(r, axis=4)
        [e, v] = tf.linalg.eig(r_avg)
        sort_idx = tf.argsort(tf.abs(e), axis=-1, direction='DESCENDING')
        v_sort = tf.gather(v, sort_idx, axis=-1, batch_dims=4)
        p_sb = v_sort[..., :self._num_layers]
        norm = tf.sqrt(tf.reduce_sum(tf.abs(p_sb) ** 2, axis=-2, keepdims=True))
        p_sb = p_sb / tf.cast(norm, p_sb.dtype) / tf.sqrt(tf.cast(self._num_layers, p_sb.dtype))
        w = tf.squeeze(p_sb[:, :, 0, ...], axis=1)
        w = rearrange(w, 'batch sb tx layer -> batch sb (tx layer)')
        w = tf.concat([tf.math.real(w), tf.math.imag(w)], axis=-1)
        # From eigenvectors to uplink information
        x = self._embedding(w)
        x += self._pos_encoding[:, :x.shape[1], :]
        x = self._dropout(x)
        for i in range(self._num_block):
            x = self._enc_layers[i](x)
        x = self._layer_norm(x)
        x = self._dense1(x)
        x = rearrange(x, 'batch sb dim -> batch (sb dim)')
        x = self._dense2(x)
        x = tf.complex(x[:, :int(x.shape[-1] / 2)] - 0.5, x[:, int(x.shape[-1] / 2):] - 0.5)

        # Scheme 2======================================================================================================
        # ...

        # Scheme 3======================================================================================================
        # ...

        # ...
        return x

class Transmitter(Layer):
    def __init__(self,
                 num_tx_ant=32,
                 embedding_dim=128,
                 num_head=8,
                 num_block=5,
                 dropout_rate=0.0,
                 num_subbands = 3,
                 num_layers = 4,
                 num_bits_per_layer = 2,
                 num_data_subcarriers = 144,
                 num_data_symbols = 1,
                 num_sc_per_sb = 48,
                 **kwargs):
        super(Transmitter, self).__init__(**kwargs)
        # Uplink
        self._num_subbands = num_subbands
        self._num_sc_per_sb = num_sc_per_sb
        self._num_layers = num_layers
        self._num_bits_per_layer = num_bits_per_layer
        self._num_tx_ant = num_tx_ant
        self._num_block = num_block
        self._dense1 = Dense(num_subbands * num_tx_ant * 2 * num_layers)
        self._embedding = linear_embedding(embedding_dim)
        self._pos_encoding = positional_encoding(1000, embedding_dim)
        self._dropout = Dropout(dropout_rate)
        self._dec_layers = [TfmBlock(num_head, embedding_dim, dropout_rate)
                           for _ in range(num_block)]
        self._layer_norm = LayerNormalization()
        self._dense2 = Dense(num_tx_ant * 2 * num_layers)

        # Downlink
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._mod_layer_1 = Dense(units=256, activation='relu')
        self._mod_layer_2 = Dense(units=2 * num_layers, activation=None)

    def call(self, bits, feedback_info, snr):
        # Controlling===================================================================================================
        # Scheme switching according to feedback_info and snr

        # Scheme 1======================================================================================================
        # From uplink information to eigenvectors
        w = tf.concat([tf.math.real(feedback_info), tf.math.imag(feedback_info)], axis=-1)
        w = self._dense1(w)
        w = rearrange(w, 'batch (sb dim) -> batch sb dim', sb=self._num_subbands)
        w = self._embedding(w)
        w += self._pos_encoding[:, :w.shape[1], :]
        w = self._dropout(w)
        for i in range(self._num_block):
            w = self._dec_layers[i](w)
        w = self._layer_norm(w)
        w = self._dense2(w)
        w = tf.complex(w[..., :int(w.shape[-1] / 2)], w[..., int(w.shape[-1] / 2):])
        w = rearrange(w, 'batch sb (layer tx) -> batch sb tx layer', layer=self._num_layers)
        norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(w)), axis=2, keepdims=True))
        w = w / tf.cast(norm, w.dtype) / tf.sqrt(tf.cast(self._num_layers, w.dtype))
        w = repeat(w , 'batch sb tx_ant layer -> batch tx ofdm (sb sc_per_sb) tx_ant layer',
                                tx=1, ofdm=1, sc_per_sb=self._num_sc_per_sb)

        # From data bits to signal
        b = bits[:,:self._num_data_subcarriers*self._num_data_symbols*self._num_layers*self._num_bits_per_layer]
        b = tf.reshape(b, [-1, 1, self._num_data_subcarriers*self._num_data_symbols, self._num_layers*self._num_bits_per_layer])
        # Modulation
        z = self._mod_layer_1(b)
        z = self._mod_layer_2(z)
        z = tf.complex(z[..., 0::2], z[..., 1::2])
        energy = tf.reduce_mean(tf.square(tf.abs(z)), axis=(1, 2, 3), keepdims=True)
        z = z / tf.cast(tf.sqrt(energy), tf.complex64)
        # Precoding
        z = tf.reshape(z, [-1, 1, self._num_data_symbols, self._num_data_subcarriers, self._num_layers, 1])
        x = tf.squeeze(tf.matmul(w, z), -1)
        x = tf.transpose(x, [0, 1, 4, 2, 3])
        x = tf.squeeze(x, axis=1)
        ctrl_bits = tf.ones([tf.shape(x)[0],5])

        # Scheme 2======================================================================================================
        # ...

        # Scheme 3======================================================================================================
        # ...

        # ...
        return x, ctrl_bits

class Receiver(Layer):
    def __init__(self,num_blocks=10,
                 num_data_subcarriers = 144,
                 num_data_symbols = 1,
                 num_tx_ant = 32,
                 num_rx_ant = 4,
                 num_layers = 4,
                 num_bits_per_layer = 2):
        super().__init__()
        self._num_data_subcarriers = num_data_subcarriers
        self._num_data_symbols = num_data_symbols
        self._num_tx_ant = num_tx_ant
        self._num_rx_ant = num_rx_ant
        self._num_blocks = num_blocks
        self._num_layers = num_layers
        self._num_bits_per_layer = num_bits_per_layer

    def build(self, input_shape):
        self.blocks = keras.models.Sequential()
        for block_id in range(self._num_blocks):
            block = ResBlock()
            self.blocks.add(block)
        self._conv_1 = layers.Conv2D(128, (1, 3), padding='SAME', activation=None)
        self._conv_2 = layers.Conv2D(self._num_layers*self._num_bits_per_layer, (1, 3), padding='SAME', activation=None)

    def call(self, y, h, ctrl_bits, snr):
        # Controlling===================================================================================================
        # Scheme switching according to ctrl_bits, h and snr

        # Scheme 1======================================================================================================
        y = tf.transpose(y, [0, 2, 3, 1])

        h = tf.reshape(h, [-1, self._num_tx_ant*self._num_rx_ant, self._num_data_symbols, self._num_data_subcarriers])
        h = tf.transpose(h, [0, 2, 3, 1])

        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       tf.math.real(h),
                       tf.math.imag(h),
                       ], axis=-1)

        z = self._conv_1(z)
        z = self.blocks(z)
        z = self._conv_2(z)
        llr = tf.reshape(z, [-1, self._num_data_symbols*self._num_data_subcarriers*self._num_layers*self._num_bits_per_layer])

        # Scheme 2======================================================================================================
        # ...

        # Scheme 3======================================================================================================
        # ...

        # ...
        return llr










