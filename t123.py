import tensorflow as tf
import numpy as np

b = tf.constant(np.arange(1, 25, dtype=np.int32),
                shape=[2, 4, 3])

a = tf.constant(np.arange(1,13), dtype=np.int32, shape=[2, 2, 3])
print(b)
print(tf.matmul(a,b,transpose_b=True))