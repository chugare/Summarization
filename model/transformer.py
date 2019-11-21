import tensorflow as tf
import numpy as np



def get_angles(pos, i, d_model):
    '''
    位置编码中的计算角度的函数
    '''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    '''
    计算位置编码的函数
    :param position: 编码的位置，位置越靠前的编码中的角度变化越小，相当于引入一个频率的信息
    :param d_model: 模型的深度也就是向量的维度
    :return:
    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    '''
        用于产生mask

    :param seq:
    :return:
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
def create_foresee_mask():
    

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """分拆最后一个维度到 (num_heads, depth).
        转置结果使得形状为 (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class TransformerGenerator:
    def __init__(self,batch_size,max_length,vocab_size,num_heads,d_model):
        self.num_heads = num_heads
        self.d_model = d_model
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.position_code = [positional_encoding(i,d_model) for i in range(1000)]

        pass

    def build_model_fn(self, input):

        def model_fn(feature, labels, mode, params=None, config=None):
            source_seqs = feature['source_seqs']
            source_pad_mask = create_padding_mask(source_seqs)
            context_seqs = feature['context_seqs']
            context_length = feature['context_length']
            context_pad_mask = create_padding_mask(context_seqs)
            target_word = feature['target_word']

            embedding_var = tf.get_variable('embedding_var',shape=[self.vocab_size, self.d_model],dtype=tf.float32,
                                            initializer=tf.truncated_normal_initializer(stddev=0.2))
            source_seqs_vec = tf.nn.embedding_lookup(embedding_var,source_seqs)+self.position_code
            context_seqs_vec = tf.nn.embedding_lookup(embedding_var,context_seqs)+self.position_code





            res_spec = tf.estimator.EstimatorSpec()