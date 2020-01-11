import tensorflow as tf
import numpy as np

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


class LSTMencoder(tf.keras.layers):
    def __init__(self,d_model, input_vocab_size,size = 128,layer_num = 3):
        self.cell = tf.keras.layers.LSTMCell(units=size,dropout=0.1)

        self.rnn_layer = tf.keras.layers.RNN(self.cell)
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)

    def call(self,source):
        source_emb = self.Embedding(source)
        state,output =  self.rnn_layer(source_emb)
        return state


class LSTMdecoder(tf.keras.layers):
    def __init__(self,d_model, input_vocab_size,size = 128):
        self.cell = tf.keras.layers.LSTMCell(units=size,dropout=0.1)

        self.rnn_layer = tf.keras.layers.RNN(self.cell)
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        pass
    def call(self,enc_vec,init_state,last_word,out_word,mode):
        input_embedding = self.Embedding(last_word)
        if mode == 'train':
            self.rnn_layer(out_word)




class S2Smodel(tf.keras.Model):

    def __init__(self):
        pass

    def call(self,source,output,label,mode):





