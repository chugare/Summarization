import tensorflow as tf
import numpy as np




def create_padding_mask(seq):
    '''
        用于产生mask

    :param seq:
    :return:
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到注意力对数（logits）。
    return seq  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class Attention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.P = tf.keras.layers.Dense(d_model)
    def call(self,x,y,mask):
        a = tf.matmul(x,self.P(y))
        if mask is not None:
            a += mask * -1e9
        a = tf.nn.softmax(a,axis=-1)
        out = tf.matmul(a,x)
        return out



class ABSEncoder(tf.keras.layers.Layer):

    def __init__(self,d_model, context_len, input_vocab_size,
                rate=0.1):
        super(ABSEncoder, self).__init__()
        self.F = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.G = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.context_len = context_len
        self.d_model = d_model
        self.attention = Attention(d_model)

    def call(self, x, yc, training, mask):
        yc = self.G(yc)
        x = self.F(x)
        batch_size, seq_len = yc.shape[:1]
        yc = tf.reshape(yc,[batch_size, -1, self.context_len * self.d_model])
        out = self.attention(x,yc,mask)
        return out

class Decoder(tf.keras.layers.Layer):
    def __init__(self,d_model, context_len, input_vocab_size,
                 hidden_size,rate=0.1):
        self.E = tf.keras.layers.Embedding(input_vocab_size,d_model)

        self.U = tf.keras.layers.Dense(hidden_size)
        self.V = tf.keras.layers.Dense(input_vocab_size)
        self.W = tf.keras.layers.Dense(input_vocab_size)

    def call(self,enc,y,mask):
        yc = self.E(y)
        batch_size, seq_len = yc.shape[:1]
        yc = tf.reshape(yc,[batch_size, -1, self.context_len * self.d_model])
        h = self.U(yc)
        p_out = self.V(h) + self.W(enc)
        return p_out

class ABS(tf.keras.Model):

    def __init__(self, d_model,context_len,input_vocab_size,hidden_size, rate=0.1):
        self.Encoder = ABSEncoder(d_model,context_len,input_vocab_size,rate)
        self.Decoder = Decoder(d_model,context_len,input_vocab_size,hidden_size,rate)

    def call(self,x,yc,mode):
        # if mode == 'TRAIN':
        mask = create_padding_mask(yc)
        enc = self.Encoder(x,yc,mode,mask)
        p_out = self.Decoder(enc,yc)
        return p_out

