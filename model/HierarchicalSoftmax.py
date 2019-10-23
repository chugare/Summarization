import tensorflow as tf
import numpy as np
import random
from data_util.data_pipe import DictFreqThreshhold,WordVec
def hierarchical_softmax(name,EncodingLength,):
    pass


batchSize = 128
vecSize = 300
maxLength = 17
input = tf.placeholder(shape=[batchSize,vecSize],dtype=tf.float32)
huffman = tf.placeholder(shape=[batchSize,maxLength],dtype=tf.int32)
length = tf.placeholder(shape=[batchSize],dtype=tf.int32)
wordid = tf.placeholder(shape=[batchSize],dtype=tf.int32)
label = tf.placeholder(shape=[batchSize,maxLength],dtype=tf.int32)

weightSet = tf.get_variable(name='Hierarchical_Weight',shape=[np.power(2,maxLength),vecSize],
                            dtype=tf.float32,initializer=tf.truncated_normal_initializer())
LossTA = tf.TensorArray(name='LossTA',size=batchSize,dtype=tf.float32)
precMicro = tf.TensorArray(name='PrecMicro',size=batchSize,dtype=tf.int32)
precMacro = tf.TensorArray(name='PrecMacro',size=batchSize,dtype=tf.int32)

def loopOpt(i,lossta,pmic,pmac):
    l = length[i]
    w = tf.gather(weightSet,huffman[i,:l])
    lab = label[i,:l]
    out = tf.tensordot(input[i],w,[0,-1])

    result = tf.cast(tf.greater(out, 0.5), tf.int32)
    precAllLevel = tf.cast(tf.equal(result,lab),tf.int32)
    precRes = tf.reduce_prod(precAllLevel)
    precAllLevel = tf.reduce_sum(precAllLevel)
    pmic = pmic.write(i,precAllLevel)
    pmac = pmac.write(i,precRes)
    lab = tf.cast(lab, tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=lab)
    loss = tf.reduce_sum(loss)
    lossta = lossta.write(i,loss)
    i = i+1
    return i,lossta,pmic,pmac
i = tf.constant(0)
i,lossTa,precMicro,precMacro = tf.while_loop(lambda i,*_: tf.less(i,batchSize),loopOpt,[i,LossTA,precMicro,precMacro])
tl = lossTa.stack()

tc = tf.reduce_sum(length)
tc = tf.cast(tc,tf.float32)
precMacro = precMacro.stack()
precMicro = precMicro.stack()
precMicro = tf.cast(precMicro,tf.float32)
precMacro = tf.cast(precMacro,tf.float32)
precMicro = tf.reduce_sum(precMicro)/(tc+1)
precMacro = tf.reduce_mean(precMacro)
tl = tf.reduce_sum(tl)
tl = tl/(tc+1)
opt = tf.train.AdamOptimizer(learning_rate=0.01)
# train = opt.minimize(tl)

DC = DictFreqThreshhold(ReadNum = 10000,DictName = "DP_comma_DICT.txt")
DC.HuffmanEncoding()
WV = WordVec(ReadNum = 10000)


weightAll = tf.get_variable(name='Normal_Weight',shape=[3000,vecSize],
                            dtype=tf.float32,initializer=tf.truncated_normal_initializer())
out = tf.tensordot(input,weightAll,[-1,-1])
loss_plain = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels= tf.one_hot(wordid,depth=3000))
loss_plain = tf.reduce_mean(loss_plain)
res = tf.cast(tf.argmax(out,axis=-1),tf.int32)
prec  = tf.cast(tf.equal(res,wordid),tf.float32)
prec = tf.reduce_mean(prec)
opt = tf.train.AdamOptimizer(learning_rate=0.01)
train = opt.minimize(loss_plain)



with tf.Session() as sess:

    bcount = 0
    vecl = []
    labell =[]
    huffl = []
    lengthl = []
    ids = []
    sess.run(tf.initialize_all_variables())
    for i in range(30):
        print('=== EPOCH %d start ==='%i)
        for k in DC.N2GRAM:
            v = WV.get_vec(DC.N2GRAM[k])
            huffman_str = DC.N2HUFF[k]
            tlen = len(huffman_str)

            tmphuff = np.zeros(shape=[maxLength],dtype=np.int32)
            tmplabel = np.zeros(shape=[maxLength],dtype=np.int32)
            currval = 0

            for i,c in enumerate(huffman_str):
                if c == '0':
                    tmphuff[i] = currval
                    tmplabel[i] = 0
                    currval = currval * 2 + 1

                if c == '1':
                    tmphuff[i] = currval
                    tmplabel[i] = 1
                    currval = currval * 2 + 2
            vecl.append(v)
            labell.append(tmplabel)
            huffl.append(tmphuff)
            lengthl.append(tlen)
            ids.append(k)
            if len(vecl) == batchSize:
                _,lossres,preout = sess.run([train,loss_plain,prec],feed_dict={
                    input:vecl,
                    length:lengthl,
                    label:labell,
                    huffman:huffl,
                    wordid:ids
                })
                #
                # _,lossres,pmac,pmic = sess.run([train,tl,precMacro,precMicro],feed_dict={
                #     input:vecl,
                #     length:lengthl,
                #     label:labell,
                #     huffman:huffl,
                #     wordid:ids
                # })
                # print("%.4f  %.3f  %.3f"%(lossres,pmac,pmic))
                print("%.4f  %.3f"%(lossres,preout))
                vecl = []
                labell = []
                huffl = []
                lengthl = []
                ids =[]
    for i in range(50):
        index = random.randint(0,len(DC.N2GRAM)-1)
        word = DC.N2GRAM[index]





