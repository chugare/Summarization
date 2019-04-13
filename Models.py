import tensorflow as tf
import numpy as np


class unionGenerator:
    def __init__(self,**kwargs):
        self.KeyWordNum = 20
        self.VecSize = 300
        self.ContextLen = 10
        self.HiddenUnit = 200
        self.TopicNum = 30
        self.FlagNum = 30
        self.TopicVec = 10
        self.FlagVec = 10
        self.ContextVec =
        self.WordNum = 50000
        self.BatchSize = 64
        self.L2NormValue = 0.02
        self.DropoutProb = 0.5
        for k,v in kwargs.items():
            self.__setattr__(k,v)
        pass
    def get_variable(self,name,shape,dtype,initializer):
        var = tf.get_variable(name,shape,dtype,initializer)
        l2_norm = tf.nn.l2_loss(var) * self.L2NormValue
        tf.add_to_collection('total loss',l2_norm)
        return var
    def build_model(self):

        keyWordVector = tf.placeholder(tf.float32,
                                       shape=[self.BatchSize,self.KeyWordNum,self.VecSize],
                                       name="Key Word Vector")
        wordVector = tf.placeholder(tf.float32,
                                       shape=[self.BatchSize,self.ContextLen,self.VecSize],
                                       name="Context Vector")
        topicSeq = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize,self.ContextLen],
                                  name="Topic Sequence")
        flagSeq = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize,self.ContextLen],
                                  name="Flag Sequence")
        topicVecMap = self.get_variable("Topic Vec Map",shape=[self.TopicNum,self.TopicVec],dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer())
        flagVecMap = self.get_variable("Flag Vec Map",shape=[self.FlagNum,self.FlagVec],dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())

        topicVec = tf.nn.embedding_lookup(topicVecMap,topicSeq)
        flagVec = tf.nn.embedding_lookup(flagVecMap,flagSeq)
        rawVecSize = self.VecSize + self.TopicVec+self.FlagVec

        encAtteWeight = self.get_variable("Encoder Attention",shape=[self.VecSize,self.ContextVec],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        encHiddenWeight = self.get_variable("Encoder Hidden 0",shape=[rawVecSize,self.ContextVec],dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer())

        contextVector = tf.concat_v2([wordVector,topicVec,flagVec],axis=-1,name="Context Vec _raw")
        contextVector = tf.tensordot(contextVector,encHiddenWeight,axes=[2,0],)
        contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
        contextVector = tf.nn.relu(contextVector,name="Context Vec")

        a = tf.tensordot(contextVector,encAtteWeight,axes=[2,1])
        a = tf.tensordot(a,keyWordVector,axes=[2,2])

        alignKeyWord

