import tensorflow as tf
import numpy as np

#
inp = tf.constant(value=1,shape=[1,100],dtype=tf.float32)
cell = tf.keras.layers.LSTMCell(200)
initstate = cell.get_initial_state(batch_size=1,dtype=tf.float32)
# initstate = [tf.zeros([1,200],tf.float32)]
layer = tf.keras.layers.RNN([cell])
print(initstate)

r = layer(inp,initial_state=initstate)
print(r)



# b = tf.constant(np.arange(1, 25, dtype=np.int32),
#                 shape=[6, 4])
# bb = tf.tile(b,[1,3])
# bbb = tf.reshape(bb,[-1,3,4])
# # tf.transpose(bbb,[0,2,1])
# a = tf.constant(np.arange(25,97), dtype=np.int32, shape=[6, 3, 4])
#
# print(bbb)
#
# ac = tf.concat([a,bbb],-1)
# print(ac)