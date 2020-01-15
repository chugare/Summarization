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


class LSTMencoder(tf.keras.layers.Layer):
    def __init__(self,d_model, input_vocab_size,size,layer_num):
        super(LSTMencoder,self).__init__()
        self.cell = []
        for i in range(layer_num):
            self.cell.append(tf.keras.layers.LSTMCell(units=size,dropout=0.1))
        self.rnn_layer = tf.keras.layers.RNN(self.cell)
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.out = tf.keras.layers.Dense(d_model)
    def call(self,source):
        source_emb = self.Embedding(source)
        output =  self.rnn_layer(source_emb)
        return self.out(output)


class LSTMdecoder(tf.keras.layers.Layer):
    def __init__(self,d_model, input_vocab_size,size,layer_num):
        super(LSTMdecoder, self).__init__()

        self.cell = tf.keras.layers.LSTMCell(units=size,dropout=0.1)

        self.rnn_layer = tf.keras.layers.RNN(self.cell,return_sequences = True,return_state=True)
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.d_model = d_model
        self.DecoderInputDense = tf.keras.layers.Dense(d_model)
        pass
    def call(self,enc_vec,init_state,last_word,label_seq,mode):
        if mode == 'train':
            label_seq_embedding = self.Embedding(label_seq)

            # 对outword使用attention或者直接使用
            seq_len = tf.shape(label_seq)[-1]
            enc_vec = tf.reshape(tf.tile(enc_vec,[1,seq_len]),[-1,seq_len,self.d_model])
            #
            enc_vec = tf.math.l2_normalize(enc_vec,-1)
            label_seq_embedding = tf.math.l2_normalize(label_seq_embedding,-1)
            decoder_input = tf.concat([enc_vec,label_seq_embedding],axis = -1)
            decoder_input = self.DecoderInputDense(decoder_input)
            # decoder_input = label_seq_embedding
            decoder_output, decoder_states_c,decoder_states_s = self.rnn_layer(decoder_input)
            # return decoder_output, [decoder_states_c,decoder_states_s]
            return decoder_output,decoder_input
            #
            # res = self.rnn_layer(decoder_input)
            # print(res)
            # return res
        else:
            last_word_embedding = self.Embedding(last_word)
            # if init_state is None:
            decoder_output,decoder_states = self.cell(last_word_embedding,init_state)
            return decoder_output,decoder_states







class RNNS2Smodel(tf.keras.Model):

    def __init__(self,d_model, input_vocab_size,encoder_size,encoder_layer_num,decoder_size,decoder_layer_num):
        super(RNNS2Smodel, self).__init__()

        self.encoder = LSTMencoder(d_model=d_model,input_vocab_size=input_vocab_size,size=encoder_size,
                                   layer_num=encoder_layer_num)
        self.decoder = LSTMdecoder(d_model=d_model,input_vocab_size=input_vocab_size,size=decoder_size,
                                   layer_num=decoder_layer_num)
        self.final_layer = tf.keras.layers.Dense(input_vocab_size)
        pass

    def call(self,source,last_word,label,mode,init_state,mask):
        enc_vec = self.encoder(source)
        # enc_vec = None
        decoder_output, decoder_state = self.decoder(enc_vec,init_state,last_word,label,mode)
        decoder_output = self.final_layer(decoder_output)
        return decoder_output,decoder_state






