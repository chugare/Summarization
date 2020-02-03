import sys,threading,queue,os
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.RNNs2s import RNNS2Smodel,create_padding_mask

import time
from interface.NewsInterface import NewsBeamsearcher,NewsPredictor


MODEL_PATH = './RNNs2s'
DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data'

D_MODEL = 200
ENCODE_SIZE = 256
ENCODER_NUM = 3
DECODER_SIZE = 256
DECODER_NUM = 3
VOCAB_SIZE = 100000

BATCH_SIZE = 10
INPUT_SEQ_LEN = 1000
OUT_SEQ_LEN = 100


class s2smodel(tf.keras.Model):
    def __init__(self):
        super(s2smodel,self).__init__()
    def call(self,features):
        source = features['source']
        last_word = features['last_word']
        title = features['title']
        model = RNNS2Smodel(d_model=D_MODEL,input_vocab_size=VOCAB_SIZE,
                            encoder_size = ENCODE_SIZE,encoder_layer_num=ENCODER_NUM,
                            decoder_size=DECODER_SIZE,decoder_layer_num=DECODER_NUM,batch_size=BATCH_SIZE)
        return model(source,last_word,title,'train',None)
def keras_train():
    pass

def build_input_fn(name,data_set,batch_size ,input_seq_len ,output_seq_len ):

    def generator():
        tokenizer = tokenization(DICT_PATH,DictSize=100000)
        source_file = queue_reader(name,data_set)
        for line in source_file:
            try:
                example = line.split('#')
                title = example[0]
                desc = example[1]
                content = example[2]
                title = title.split(' ')
                content = content.split(' ')
                source_sequence = tokenizer.tokenize(content)
                source_sequence = tokenizer.padding(source_sequence,input_seq_len)
                title_sequence = tokenizer.tokenize(title)
                title_sequence = tokenizer.padding(title_sequence,output_seq_len)
                title_sequence.insert(0,2)

                # title_context.append([0]*context_le)

                feature = {
                    'source':source_sequence,
                    'last_word': 0,
                    'title':title_sequence[:-1],
                }
                label = title_sequence[1:]
                #     yield feature,label
                yield feature,label
            except Exception:
                continue
    def input_fn():
        ds = tf.data.Dataset.from_generator(generator=generator,output_types=({'source':tf.int64,'last_word':tf.int64,'title':tf.int64},tf.int64),output_shapes=({'source':[input_seq_len],'last_word':[],'title':[output_seq_len]},[output_seq_len]))
        ds = ds.shuffle(8192).batch(batch_size,drop_remainder=True)
        return ds
    return input_fn



train_dataset = build_input_fn("NEWS", DATA_PATH,batch_size=BATCH_SIZE,input_seq_len=INPUT_SEQ_LEN,output_seq_len=OUT_SEQ_LEN)()

print(train_dataset)
checkpoint_dir = './rnn_training_checkpoints'

# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
model = s2smodel()
model.compile(optimizer='adam', loss=loss)
# model.build(input_shape={'source':[BATCH_SIZE,INPUT_SEQ_LEN],'last_word':[BATCH_SIZE],'title':[BATCH_SIZE,OUT_SEQ_LEN]})
history = model.fit(train_dataset,epochs=10,callbacks=[checkpoint_callback])
# model.summary()