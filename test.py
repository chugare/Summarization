import jieba
import tensorflow as tf
import sys
from jieba import posseg
import  numpy as np
def w():
    writer = tf.python_io.TFRecordWriter('DP_TFR.tfrecord')

    raw = [[range(i,i+5)] for i in range(10)]


    raw = np.array(raw,dtype=np.float32)
    print(raw.ndim)
    scal = 1.0
    print(np.isscalar(scal))
    label  = 13
    raw = np.reshape(raw,[-1])
    print(raw)
    example = get_feature(f1=raw,f2=scal,f3 = label)
    for i in range(1000):
        writer.write(example.SerializeToString())
    writer.close()

def get_feature(**kwargs):
    features = {}
    for k in kwargs:
        var = kwargs[k]
        if not np.isscalar(var):
            var = np.reshape(var,[-1])
        else:
            var = np.array([var])
        if var.dtype == np.float32 or var.dtype == np.float64:
            features[k] = tf.train.Feature(float_list = tf.train.FloatList(value = var))
        elif var.dtype == np.int32 or var.dtype == np.int64:
            features[k] = tf.train.Feature(int64_list = tf.train.Int64List(value = var))
        else:
            features[k] = tf.train.Feature(bytes_list = tf.train.BytesList(value = var))
    example = tf.train.Example(features=tf.train.Features(
        feature=features
    ))
    return example

def r():

    reader = tf.TFRecordReader()
    fnq = tf.train.string_input_producer(['DP_TFR.tfrecord'])
    i,ser_example = reader.read(fnq)
    features = tf.parse_single_example(ser_example,features={
        'f1':tf.FixedLenFeature([5*10],tf.float32),
        'f2':tf.FixedLenFeature([],tf.float32),
        'f3':tf.FixedLenFeature([],tf.int64),
    })
    features =tf.train.shuffle_batch(features,batch_size=64,
                                     capacity=20000,
                                     num_threads=4,
                                     min_after_dequeue=10000)
    f1 = features['f1']
    f1 = tf.reshape(f1,[64,5,10])
    f2 = features['f2']
    f3 = features['f3']
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        fr1,fr2,fr3 = sess.run([f1,f2,f3])
        print(fr1)
        print(fr2)
        print(fr3)
        coord.request_stop()
        coord.join(threads)
def kk(**kwargs):
    return kwargs
# r()
#

m = {1:'p',
     2:'e',
     }
print(list(m.values())[0])

# var1 = tf.placeholder(shape=[50],dtype=tf.float32)
# val = np.arange(50)
# vart = tf.reshape(var1,[5,2,5])
# var2 = tf.ones(shape=[5,2,5])
# res = tf.reduce_prod(var2)
# res2 = var2*20
# res2 = tf.greater(vart,res2)
# res2 = tf.cast(res2,tf.float32)
# with tf.Session() as sess:
#     p = sess.run(res,feed_dict={
#         var1:val
#     })
#     print(p)
#
#     p = sess.run(res2,feed_dict={
#         var1:val
#     })
#     print(p)
