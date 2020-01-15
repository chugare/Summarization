import tensorflow as tf
import numpy as np

b = tf.constant(np.arange(1, 25, dtype=np.int32),
                shape=[6, 4])
bb = tf.tile(b,[1,3])
bbb = tf.reshape(bb,[-1,3,4])
# tf.transpose(bbb,[0,2,1])
a = tf.constant(np.arange(25,97), dtype=np.int32, shape=[6, 3, 4])

print(bbb)

ac = tf.concat([a,bbb],-1)
print(ac)