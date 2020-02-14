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
        a = tf.matmul(x,self.P(y),transpose_b=True)
        if mask is not None:

            a += mask * -1e9
        print(a)
        a = tf.nn.softmax(a,axis=-2)

        out = tf.matmul(a,x,transpose_a=True)
        print(out)
        return out


class LSTMencoder(tf.keras.layers.Layer):
    def __init__(self,d_model, input_vocab_size,size,layer_num):
        super(LSTMencoder,self).__init__()
        self.cell = []
        for i in range(layer_num):
            self.cell.append(tf.keras.layers.LSTMCell(units=size,dropout=0.1))
        self.rnn_layer = tf.keras.layers.RNN(self.cell,return_sequences=True)
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.out = tf.keras.layers.Dense(d_model)

    def call(self,source,source_len):
        source_emb = self.Embedding(source)
        output =  self.rnn_layer(source_emb)
        # batch_size = tf.shape(source_len)[0]
        # index = tf.range(0,batch_size,dtype=tf.int64)
        # index = tf.expand_dims(index,-1)
        # source_len = tf.expand_dims(source_len,-1)
        # index = tf.concat([index,source_len],-1)
        # # index = tf.split(index,batch_size)
        # output = tf.gather_nd(output,index)

        return output,source_emb


class LSTMdecoder(tf.keras.layers.Layer):
    def __init__(self,d_model, input_vocab_size,size,layer_num):
        super(LSTMdecoder, self).__init__()
        self.cell = []
        self.size = size
        for i in range(layer_num):
            self.cell.append(tf.keras.layers.LSTMCell(units=size,dropout=0.1))

        # self.cell = tf.keras.layers.LSTMCell(units=size,dropout=0.1)

        self.rnn_layer = tf.keras.layers.RNN(self.cell,return_sequences = True,return_state=True)
        self.Embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)
        self.d_model = d_model
        self.DecoderInputDense = tf.keras.layers.Dense(d_model)
        self.attention = Attention(size)
        pass
    def call(self,enc_vec,init_state,last_word,mode,mask):

        last_word_embedding = self.Embedding(last_word)
        # last_word_embedding = tf.reshape(last_word_embedding,[-1,1,self.d_model])
        # enc_vec = tf.reshape(enc_vec,[-1,1,self.size])
        # enc_vec = tf.math.l2_normalize(enc_vec,-1)
        decoder_input = self.attention(enc_vec,last_word_embedding,mask)
        decoder_input = self.DecoderInputDense(decoder_input)

        if mode == 'train':
            res = self.rnn_layer(decoder_input)
            decoder_output = res[0]
            return decoder_output,tf.constant(0)
            #
        else:
            res = self.rnn_layer(decoder_input,init_state)
            decoder_output,decoder_states = res[0],res[1:]
            states = []
            for s in decoder_states:
                states.extend([tf.expand_dims(cstate,1) for cstate in s])

            states = tf.concat(states,1)
            # return decoder_output,[decoder_states_c,decoder_states_m]
            return decoder_output,states





class RNNS2Smodel(tf.keras.Model):

    def __init__(self,d_model, input_vocab_size,encoder_size,encoder_layer_num,decoder_size,decoder_layer_num):
        super(RNNS2Smodel, self).__init__()

        self.encoder = LSTMencoder(d_model=d_model,input_vocab_size=input_vocab_size,size=encoder_size,
                                   layer_num=encoder_layer_num)
        self.decoder = LSTMdecoder(d_model=d_model,input_vocab_size=input_vocab_size,size=decoder_size,
                                   layer_num=decoder_layer_num)
        self.final_layer = tf.keras.layers.Dense(input_vocab_size)
        pass

    def call(self,source,source_len,last_word,mode,init_state,mask=None):
        enc_vec,enc_rnn_out = self.encoder(source,source_len)
        # enc_vec = None
        mask = create_padding_mask(source)
        mask = tf.expand_dims(mask,-1)
        seq_len = tf.shape(last_word)[1]
        mask = tf.tile(mask,[1,1,seq_len])
        decoder_output, decoder_state = self.decoder(enc_vec,init_state,last_word,mode,mask)
        decoder_output = self.final_layer(decoder_output)

        return decoder_output,decoder_state,enc_vec
    def get_initial_state(self,input):
        return self.decoder.rnn_layer.get_initial_state()





