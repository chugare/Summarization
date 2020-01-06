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
        a = tf.matmul(x,self.P(y),transpose_b = True)

        if mask is not None:
            a += mask * -1e9
        a = tf.transpose(a,[0,2,1])
        a = tf.nn.softmax(a,axis=-1)

        out = tf.matmul(a,x)
        return out,a



class ABSEncoder(tf.keras.layers.Layer):

    def __init__(self,d_model, context_len, input_vocab_size,seq_len,
                rate=0.1):
        super(ABSEncoder, self).__init__()
        self.F = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.G = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.context_len = context_len
        self.d_model = d_model
        self.seq_len = seq_len
        self.attention = Attention(d_model)

    def call(self, x, yc, training,mask):
        yc = self.G(yc)
        mask = create_padding_mask(x)
        mask = tf.tile(tf.expand_dims(mask,-1),[1,1,self.seq_len])
        x = self.F(x)

        yc = tf.reshape(yc,[-1, self.seq_len , self.context_len * self.d_model])
        # mask = tf.tile(tf.expand_dims(mask,-1),[1,1,1,self.d_model])
        # mask = tf.reshape(mask,[-1, self.seq_len , self.context_len * self.d_model])
        out,attention_w= self.attention(x,yc,mask)
        return out,attention_w

class Decoder(tf.keras.layers.Layer):
    def __init__(self,d_model, seq_len,context_len, input_vocab_size,
                 hidden_size,rate=0.1):
        super(Decoder,self).__init__()

        self.E = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.seq_len = seq_len
        self.context_len = context_len
        self.d_model = d_model
        self.U = tf.keras.layers.Dense(hidden_size)
        self.V = tf.keras.layers.Dense(input_vocab_size)
        self.W = tf.keras.layers.Dense(input_vocab_size)

    def call(self,enc,y):
        yc = self.E(y)
        yc = tf.reshape(yc,[-1, self.seq_len, self.context_len * self.d_model])
        h = self.U(yc)
        p_out = self.V(h) + self.W(enc)
        return p_out

class ABS(tf.keras.Model):

    def __init__(self, d_model,context_len,seq_len,input_vocab_size,hidden_size, rate=0.1):
        super(ABS,self).__init__()
        self.Encoder = ABSEncoder(d_model,context_len,input_vocab_size,seq_len,rate)
        self.Decoder = Decoder(d_model,seq_len,context_len,input_vocab_size,hidden_size,rate)

    def call(self,x,yc,mask,mode):
        # if mode == 'TRAIN':

        enc, attention_w = self.Encoder(x,yc,mode,mask)
        p_out = self.Decoder(enc,yc)
        return p_out,attention_w

