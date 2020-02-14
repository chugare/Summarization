
# #
# inp = tf.constant(value=1,shape=[10,1,100],dtype=tf.float32)
# cell = []
# for i in range(5):
#     cell.append(tf.keras.layers.LSTMCell(200))
# # initstate  cell.get_initial_state(batch_size=1,dtype=tf.float32)
# initstate = tf.zeros([10*5*2,200],tf.float32)
# initstate = tf.split(initstate,5)
# ist = []
# for i in initstate:
#     ist.append(tf.split(i,2))
#
# layer = tf.keras.layers.RNN(cell)
# initstate = layer.get_initial_state(inp)
# # print(ist)
# print(initstate)
# r = layer(inp,initial_state=ist)
# print(r)

import numpy as np

a = np.ones([100])*2
b = np.ones([100])*2
print(a*b)

#
# a = tf.range(100)
#
# aa = tf.reshape(a,[10,10])
#
# print(aa)
#
# alist = tf.split(aa,5)
#
# print(alist)
#
# ar = tf.concat(alist,0)
# print(ar)
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