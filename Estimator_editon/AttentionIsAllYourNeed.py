import tensorflow as tf

from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.transformer import Transformer,create_look_ahead_mask,create_padding_mask


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

def build_input_fn():

    def generator():
        tokenizer = tokenization("./news_data/NEWS_DICT.txt",DictSize=100000)
        source_file = queue_reader("NEWS","/home/user/zsm/Summarization/news_data")
        for line in source_file:
            example = line.split('#')
            title = example[0]
            desc = example[1]
            content = example[2]

            source_sequence = tokenizer.tokenize(content)
            source_sequence = tokenizer.padding(source_sequence,1000)
            title_sequence = tokenizer.tokenize(title)
            title_sequence = tokenizer.padding(title_sequence,100)
            # for i,s in enumerate(title_sequence):
            #     label = s
            #     context = title_sequence[:i]
            #
            #     feature = {
            #         'source':source_sequence,
            #         'context':context
            #     }
            #     yield feature,label
            yield source_sequence,title_sequence

    def input_fn():
        gen = generator()
        ds = tf.data.Dataset().from_generator(gen)
        return ds




def build_model_fn():

    learning_rate = 0.01

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def model_fn(features,labels,mode,params):

        source = features['source']
        context = features['context']
        tar_inp = context[:, :-1]
        tar_real = context[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(source, context)

        transformer = Transformer(num_layers= 3,
                            d_model=100,
                            num_heads=4,
                            dff=2,
                            input_vocab_size=100000,
                            target_vocab_size=100000,
                            pe_input=1000,pe_target=100)
        training = mode == tf.estimator.ModeKeys.TRAIN
        with tf.GradientTape() as tape:
            prediction,atte_weight = transformer(source,tar_inp,training, enc_padding_mask,
                  combined_mask, dec_padding_mask)

            loss = loss_function(tar_real,prediction)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                             epsilon=1e-9)
        gradients = tape.gradient(loss,transformer.trainable_variables)
        train_op = optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, prediction)
        return tf.estimator.EstimatorSpec(mode,prediction,loss,train_op)

    return model_fn










class AttentionIsAllYourNeed(tf.estimator.Estimator):

    def __init__(self):
        mfn = build_model_fn()
        super(AttentionIsAllYourNeed,self).__init__(mfn,"./transformer")


if __name__ == '__main__':






