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
        self.GlobalNorm = 0.5
        self.LearningRate = 0.001
        for k,v in kwargs.items():
            self.__setattr__(k,v)
        pass
    def get_meta(self):
        return self.__dict__
    def get_variable(self,name,shape,dtype,initializer):
        var = tf.get_variable(name,shape,dtype,initializer)
        l2_norm = tf.nn.l2_loss(var) * self.L2NormValue
        tf.add_to_collection('l2norm',l2_norm)
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
        selLabel = tf.placeholder(tf.float32,
                                   shape=[self.BatchSize],
                                   name="Sel_Label")
        selWordLabel = tf.placeholder(tf.int32,
                                  shape=[self.BatchSize],
                                  name="WordSel_Label")

        topicLabel = tf.one_hot(topicLabel,depth=self.TopicNum)
        flagLabel = tf.one_hot(flagLabel,depth=self.FlagNum)
        wordLabel = tf.one_hot(wordLabel,depth=self.WordNum)
        selWordLabel = tf.one_hot(selWordLabel,depth=self.KeyWordNum)

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

        contextVector = tf.concat([wordVector,topicVec,flagVec],axis=-1,name="Context_Vec_raw")
        contextVector = tf.reshape(contextVector,[self.BatchSize,-1],name="Flat_context")
        contextVector = tf.matmul(contextVector,encHiddenWeight)

        contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
        contextVector = tf.nn.relu(contextVector,name="Context_Vec")

        def attention(q,k,w):
            a = tf.matmul(q,w,transpose_b=True)
            a = tf.expand_dims(a,-1)
            align = tf.matmul(k,a)
            align = tf.nn.softmax(align,axis=1)
            return align,tf.reduce_mean(align * k,1)

        _,alignKeyWord = attention(contextVector,keyWordVector,encAtteWeight)
        selVector,_ = attention(contextVector,keyWordVector,selAtteWeight)
        selVector = tf.squeeze(selVector)
        encVector = tf.concat([alignKeyWord,contextVector],axis=-1,name = "Encoder_Vector")
        WeightTopic = self.get_variable(name="Weight_Topic",shape=[self.VecSize+self.ContextVec,self.TopicNum],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightSel = self.get_variable(name="Weight_Sel", shape=[self.VecSize + self.ContextVec,1],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightFlag = self.get_variable(name="Weight_Flag",shape=[self.VecSize+self.ContextVec,self.FlagNum],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightWord = self.get_variable(name="Weight_Word", shape=[self.VecSize + self.ContextVec,self.WordNum], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())

        topicOut = tf.matmul(encVector,WeightTopic,name="Topic_Out")
        topicOut = tf.nn.dropout(x=topicOut,keep_prob=self.DropoutProb)
        selOut = tf.matmul(encVector, WeightSel, name="Sel_Out")
        selOut = tf.reshape(selOut,shape=[self.BatchSize])
        selOut = tf.nn.dropout(x=selOut,keep_prob=self.DropoutProb)
        flagOut = tf.matmul(encVector,WeightFlag,name="Flag_Out")
        flagOut = tf.nn.dropout(x=flagOut,keep_prob=self.DropoutProb)
        wordOut = tf.matmul(encVector,WeightWord,name="Word_Out")
        wordOut = tf.nn.dropout(x=wordOut,keep_prob=self.DropoutProb)

        train = tf.no_op()
        loss = tf.no_op()

        print(selWordLabel)
        print(selVector)
        if(mode == 'train'):
            selRes = tf.nn.sigmoid_cross_entropy_with_logits(logits=selOut,labels=selLabel,name="Select_Result")
            topicRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=topicOut,labels=topicLabel,name="Topic_Result")
            flagRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flagOut,labels=flagLabel,name="Flag_Result")
            wordRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=wordOut,labels=wordLabel,name="Word_Result")
            selMap = tf.nn.softmax_cross_entropy_with_logits_v2(logits=selVector,labels=selWordLabel,name="Select_Map_Result")

            lossesTensor = [selRes,topicRes,flagRes,wordRes,selMap]
            for l in lossesTensor:
                tf.summary.scalar(name=l.name,tensor=l)
            loss = selRes + (1-tf.sigmoid(selOut))*(topicRes+flagRes+wordRes) +tf.sigmoid(selOut)*(selMap)
            loss = tf.reduce_mean(loss)
            opt = tf.train.AdamOptimizer(learning_rate=self.LearningRate)
            grads = opt.compute_gradients(loss)
            for grad,var in grads:
                tf.summary.histogram(var.name+'/gradient',grad)

            grad,var = zip(*grads)
            tf.clip_by_global_norm(grad,self.GlobalNorm)
            grads = zip(grad,var)
            train = opt.apply_gradients(grads)
        merge = tf.summary.merge_all()
        ops = {
            'keyWordVector':keyWordVector,
            'wordVector':wordVector,
            'topicSeq':topicSeq,
            'flagSeq':flagSeq,
            'topicLabel':topicLabel,
            'flagLabel':flagLabel,
            'wordLabel':wordLabel,
            'selLabel':selLabel,
            'selWordLabel':selWordLabel,
            'train':train,
            'loss':loss,
            'merge':merge
        }
        return ops
    def build_model_pipe(self,mode,input):
        # assert isinstance(input,tf.train.Features)
        if mode == 'train':
            keyWordVector = input['keyWordVector']
            keyWordVector = tf.reshape(keyWordVector,[self.BatchSize,self.KeyWordNum,self.VecSize],name='KeyWordVector')
            wordVector = input['wordVector']
            wordVector = tf.reshape(wordVector,[self.BatchSize,self.ContextLen,self.VecSize],name='WordVector')
            topicSeq = input['topicSeq']
            flagSeq = input['flagSeq']
            topicLabel = input['topicLabel']
            flagLabel = input['flagLabel']
            wordLabel = input['wordLabel']
            selWordLabel = input['selWordLabel']
            selLabel = input['selLabel']
            selLabel = tf.cast(selLabel,tf.float32)
            topicLabel = tf.one_hot(topicLabel,depth=self.TopicNum)
            flagLabel = tf.one_hot(flagLabel,depth=self.FlagNum)
            wordLabel = tf.one_hot(wordLabel,depth=self.WordNum)
            selWordLabel = tf.one_hot(selWordLabel,depth=self.KeyWordNum)

        else:
            keyWordVector = tf.placeholder(tf.float32,
                                           shape=[1,self.KeyWordNum,self.VecSize],
                                           name="Key_Word_Vector")
            wordVector = tf.placeholder(tf.float32,
                                        shape=[1,self.ContextLen,self.VecSize],
                                        name="Context_Vector")
            topicSeq = tf.placeholder(tf.int32,
                                      shape=[1,self.ContextLen],
                                      name="Topic_Sequence")
            flagSeq = tf.placeholder(tf.int32,
                                     shape=[1,self.ContextLen],
                                     name="Flag_Sequence")


        topicVecMap = self.get_variable("Topic_Vec_Map",shape=[self.TopicNum,self.TopicVec],dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer())
        flagVecMap = self.get_variable("Flag_Vec_Map",shape=[self.FlagNum,self.FlagVec],dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())
        zeros = tf.zeros(shape=[1,self.TopicVec])
        topicVecMap = tf.concat([topicVecMap, zeros], axis=0)
        zeros = tf.zeros(shape=[1, self.FlagVec])
        flagVecMap = tf.concat([flagVecMap,zeros],axis=0)

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

        contextVector = tf.concat([wordVector,topicVec,flagVec],axis=-1,name="Context_Vec_raw")
        contextVector = tf.reshape(contextVector,[self.BatchSize,-1],name="Flat_context")
        contextVector = tf.matmul(contextVector,encHiddenWeight)

        contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
        contextVector = tf.nn.relu(contextVector,name="Context_Vec")

        def attention(q,k,w):
            a = tf.matmul(q,w,transpose_b=True)
            a = tf.expand_dims(a,-1)
            align = tf.matmul(k,a)
            align = tf.nn.softmax(align,axis=1)
            return align,tf.reduce_mean(align * k,1)

        _,alignKeyWord = attention(contextVector,keyWordVector,encAtteWeight)
        selVector,_ = attention(contextVector,keyWordVector,selAtteWeight)
        selVector = tf.squeeze(selVector)
        encVector = tf.concat([alignKeyWord,contextVector],axis=-1,name = "Encoder_Vector")
        WeightTopic = self.get_variable(name="Weight_Topic",shape=[self.VecSize+self.ContextVec,self.TopicNum],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightSel = self.get_variable(name="Weight_Sel", shape=[self.VecSize + self.ContextVec,1],
                                        dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightFlag = self.get_variable(name="Weight_Flag",shape=[self.VecSize+self.ContextVec,self.FlagNum],dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())
        WeightWord = self.get_variable(name="Weight_Word", shape=[self.VecSize + self.ContextVec,self.WordNum], dtype=tf.float32,
                                        initializer=tf.truncated_normal_initializer())

        topicOut = tf.matmul(encVector,WeightTopic,name="Topic_Out")
        # topicOut = tf.nn.dropout(x=topicOut,keep_prob=self.DropoutProb)
        selOut = tf.matmul(encVector, WeightSel, name="Sel_Out")
        selOut = tf.reshape(selOut,shape=[self.BatchSize])
        # selOut = tf.nn.dropout(x=selOut,keep_prob=self.DropoutProb)
        flagOut = tf.matmul(encVector,WeightFlag,name="Flag_Out")
        # flagOut = tf.nn.dropout(x=flagOut,keep_prob=self.DropoutProb)
        wordOut = tf.matmul(encVector,WeightWord,name="Word_Out")
        # wordOut = tf.nn.dropout(x=wordOut,keep_prob=self.DropoutProb)

        train = tf.no_op()
        loss = tf.no_op()

        ops = {}
        if(mode == 'train'):
            selRes = tf.nn.sigmoid_cross_entropy_with_logits(logits=selOut,labels=selLabel)
            topicRes = (1-selLabel)*tf.nn.softmax_cross_entropy_with_logits_v2(logits=topicOut,labels=topicLabel)
            flagRes = (1-selLabel)*tf.nn.softmax_cross_entropy_with_logits_v2(logits=flagOut,labels=flagLabel)
            wordRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=wordOut,labels=wordLabel)
            selMap = selLabel*tf.nn.softmax_cross_entropy_with_logits_v2(logits=selVector,labels=selWordLabel)

            selCount = tf.reduce_sum(selLabel)
            selRes = tf.reduce_mean(selRes,name="Select_Result")
            topicRes = tf.identity(tf.reduce_sum(topicRes)/(self.BatchSize-selCount+1),name="Topic_Result")
            flagRes = tf.identity(tf.reduce_sum(flagRes)/(self.BatchSize-selCount+1),name="Flag_Result")
            wordRes = tf.identity(tf.reduce_sum(wordRes)/(self.BatchSize-selCount+1),name="Word_Result")
            selMap = tf.identity(tf.reduce_sum(selMap)/(selCount+1),name="Select_Map_Result")


            lossesTensor = [selRes,topicRes,flagRes,wordRes,selMap]
            for l in lossesTensor:
                 tf.summary.scalar(name=l.name,tensor=tf.reduce_mean(l))

            loss = selMap+topicRes+flagRes+wordRes+selRes
            omega = tf.add_n(tf.get_collection('l2norm'))
            loss = loss + omega
            tf.summary.scalar('Loss',loss)
            opt = tf.train.AdamOptimizer(learning_rate=self.LearningRate)
            grads = opt.compute_gradients(loss)
            for grad,var in grads:
                tf.summary.histogram(var.name+'/gradient',grad)

            grad,var = zip(*grads)
            tf.clip_by_global_norm(grad,self.GlobalNorm)
            grads = zip(grad,var)
            train = opt.apply_gradients(grads)
        else:
            selRes = tf.argmax(selOut,-1)
            topicRes = tf.nn.softmax(topicOut)
            flagRes = tf.nn.softmax(flagOut)
            wordRes = tf.nn.softmax(wordOut)
            selMap = tf.argmax(selVector)
            ops['keyWordVector'] = keyWordVector
            ops['wordVector'] = wordVector
            ops['topicSeq'] = topicSeq
            ops['flagSeq'] = flagSeq

            ops['wordRes'] = wordRes
            ops['topicRes'] = topicRes
            ops['flagRes'] = flagRes
            ops['selRes'] = selRes
            ops['selMap'] = selMap

        merge = tf.summary.merge_all()
        ops['train'] = train
        ops['merge'] = merge
        ops['loss'] = loss

        return ops