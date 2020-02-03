
import sys,threading,queue
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
                    'title':title_sequence,
                }
                #     yield feature,label
                yield feature
            except Exception:
                continue
    def input_fn():
        ds = tf.data.Dataset.from_generator(generator=generator,output_types={'source':tf.int64,'last_word':tf.int64,'title':tf.int64},output_shapes={'source':[input_seq_len],'last_word':[],'title':[output_seq_len+1]})
        ds = ds.shuffle(8192).batch(batch_size).cache().repeat()
        return ds
    return input_fn

def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask



def accuracy_function(real,pred):



    train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=train_accuracy.dtype)

    count = tf.reduce_sum(mask)+1
    train_accuracy = tf.reduce_sum(train_accuracy * mask) / count
    return train_accuracy




def train(EPOCHS):

    train_dataset = build_input_fn("NEWS", DATA_PATH,batch_size=BATCH_SIZE,input_seq_len=INPUT_SEQ_LEN,output_seq_len=OUT_SEQ_LEN)()

    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()

            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)

            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step,tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)

            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    learning_rate = CustomSchedule(D_MODEL)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.98,
    #                                              epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    RNNS2S = RNNS2Smodel(d_model=D_MODEL,input_vocab_size=VOCAB_SIZE,
                               encoder_size = ENCODE_SIZE,encoder_layer_num=ENCODER_NUM,
                               decoder_size=DECODER_SIZE,decoder_layer_num=DECODER_NUM)
    checkpoint_path = "./checkpoints/RNN"

    ckpt = tf.train.Checkpoint(RNNS2S=RNNS2S,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # 如果检查点存在，则恢复最新的检查点。
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    def train_step(features):

        # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        #
        source = features['source']
        last_word = features['last_word']

        title = features['title']
        title_real = title[:,1:]
        title_input = title[:,:-1]
        mask = create_padding_mask(title_real)

        with tf.GradientTape() as tape:
            prediction = RNNS2S(source,title_input,title_real,'train',None)
            prediction = prediction - tf.expand_dims(tf.reduce_max(prediction,-1),-1)
            loss = loss_function(title_real, prediction)

        gradients = tape.gradient(loss, RNNS2S.trainable_variables)
        optimizer.apply_gradients(zip(gradients, RNNS2S.trainable_variables))

        train_loss(loss)
        train_accuracy(title_real, prediction)

    for epoch in range(EPOCHS):
        start = time.time()
        lasttime = start

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, features) in enumerate(train_dataset):
            train_step(features)

            if batch % 1 == 0:
                nowtime = time.time()
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f} TCPB {:.2f}(s) TTC {:.4f}(h)'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result(),nowtime-lasttime,(nowtime-start)/3600))
                lasttime = nowtime

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                 ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                             train_loss.result(),
                                                             train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


train(10)