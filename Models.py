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
        self.ContextVec = 400
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
    def build_model(self,mode):

        keyWordVector = tf.placeholder(tf.float32,
                                       shape=[self.BatchSize,self.KeyWordNum,self.VecSize],
                                       name="Key_Word_Vector")
        wordVector = tf.placeholder(tf.float32,
                                       shape=[self.BatchSize,self.ContextLen,self.VecSize],
                                       name="Context_Vector")
        topicSeq = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize,self.ContextLen],
                                  name="Topic_Sequence")
        flagSeq = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize,self.ContextLen],
                                  name="Flag_Sequence")
        topicLabel = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize],
                                  name="Topic_Label")
        flagLabel = tf.placeholder(tf.int32,
                                    shape=[self.BatchSize],
                                    name="Flag_Label")
        wordLabel = tf.placeholder(tf.int32,
                                   shape=[self.BatchSize],
                                   name="Word_Label")
        selLabel = tf.placeholder(tf.int32,
                                   shape=[self.BatchSize],
                                   name="Sel_Label")
        selWordLabel = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize],
                                  name="WordSel_Label")
        topicVecMap = self.get_variable("Topic_Vec_Map",shape=[self.TopicNum,self.TopicVec],dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer())
        flagVecMap = self.get_variable("Flag_Vec_Map",shape=[self.FlagNum,self.FlagVec],dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())

        topicVec = tf.nn.embedding_lookup(topicVecMap,topicSeq)
        flagVec = tf.nn.embedding_lookup(flagVecMap,flagSeq)
        rawVecSize = self.VecSize + self.TopicVec+self.FlagVec
        rawVecSize *= self.ContextLen
        encAtteWeight = self.get_variable("Encoder_Attention",shape=[self.VecSize,self.ContextVec],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        selAtteWeight = self.get_variable("Select_Attention",shape=[self.VecSize,self.ContextVec],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        encHiddenWeight = self.get_variable("Encoder_Hidden_0",shape=[rawVecSize,self.ContextVec],dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer())

        print(topicVec)
        print(flagVec)
        contextVector = tf.concat([wordVector,topicVec,flagVec],axis=-1,name="Context_Vec_raw")
        print(contextVector)
        contextVector = tf.reshape(contextVector,[self.BatchSize,-1],name="Flat_context")
        contextVector = tf.matmul(contextVector,encHiddenWeight)
        print(contextVector)

        contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
        contextVector = tf.nn.relu(contextVector,name="Context_Vec")

        a = tf.matmul(contextVector,encAtteWeight,transpose_b=True)
        a = tf.expand_dims(a,-1)
        # a = tf.tensordot(a,keyWordVector,axes=[2,2])
        print(a)
        print(keyWordVector)

        align = tf.matmul(keyWordVector,a)
        # align = tf.squeeze(align)
        align = tf.nn.softmax(align,axis=1)
        print(align)
        alignKeyWord = tf.reduce_mean(align * keyWordVector,1)
        print(alignKeyWord)
        encVector = tf.concat([alignKeyWord,contextVector],axis=-1,name = "Encoder_Vector")
        WeightTopic = self.get_variable(name="Weight_Topic",shape=[self.VecSize+self.ContextVec,self.TopicNum],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightSel = self.get_variable(name="Weight_Sel", shape=[self.VecSize + self.ContextVec],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightFlag = self.get_variable(name="Weight_Flag",shape=[self.VecSize+self.ContextVec,self.FlagNum],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightWord = self.get_variable(name="Weight_Word", shape=[self.VecSize + self.WordNum], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())

        topicOut = tf.matmul(encVector,WeightTopic,name="Topic_Out")
        selOut = tf.matmul(encVector, WeightSel, name="Sel_Out")

        flagOut = tf.matmul(encVector,WeightFlag,name="Flag_Out")
        wordOut = tf.matmul(encVector,WeightWord,name="Word_Out")

        if(mode == 'train'):
            selRes = tf.nn.sigmoid_cross_entropy_with_logits()

t = unionGenerator()
t.build_model()