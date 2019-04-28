import tensorflow as tf
import numpy as np


class unionGenerator:
    def __init__(self,**kwargs):
        self.KeyWordNum = 20
        self.VecSize = 300
        self.ContextLen = 10
        self.HiddenUnit = 800
        self.KernelSize = 5
        self.KernelNum = 800
        self.TopicNum = 30
        self.FlagNum = 60
        self.TopicVec = 10
        self.FlagVec = 20
        self.ContextVec = 400
        self.WordNum = 80000
        self.BatchSize = 128
        self.L2NormValue = 0.02
        self.DropoutProb = 0.7
        self.GlobalNorm = 0.5
        self.LearningRate = 0.001
        self.HidderLayer = 3
        self.LRDecayRate = 0.8
        for k,v in kwargs.items():
            self.__setattr__(k,v)
        pass
    def get_meta(self):
        return self.__dict__
    def get_variable(self,name,shape,dtype,initializer=None):
        if initializer == None:
            initializer = tf.truncated_normal_initializer(stddev=0.2)
        var = tf.get_variable(name,shape,dtype,initializer)
        l2_norm = tf.reduce_mean(var**2) * self.L2NormValue
        tf.add_to_collection('l2norm',l2_norm)
        return var
    def get_cnn_layer(self,input_vector,name,kernelNum):
        with tf.name_scope(name):
            inShape = input_vector.shape
            kernelFilter = self.get_variable(name='Kernel_%s'%name, shape=[self.KernelSize, inShape[-2], 1, kernelNum],
                                             dtype=tf.float32)

            conv = tf.nn.conv2d(input_vector, kernelFilter,name='Convolution_%s'%name, strides=[1, 1, 1, 1],
                                padding="VALID",
                                use_cudnn_on_gpu=True)

            pool = tf.nn.max_pool(conv, ksize=[1,conv.shape[1], 1, 1],name='Pool_%s'%name,strides=[1, 1, 1, 1],
                                      padding='VALID')
            pool = tf.squeeze(pool,[1,2])
            res = tf.nn.l2_normalize(pool, axis=-1)
        return res

    def get_layer(self,name,unit,input):
        v= input.shape[-1]
        Weight = self.get_variable(name=name,shape=[v, unit], dtype=tf.float32)
        Bias = tf.get_variable(name=name+'_bias',shape=[unit])
        Out = tf.matmul(input,Weight)+Bias
        Out = tf.nn.dropout(Out,self.DropoutProb)
        Out = tf.nn.tanh(Out)
        return Out

    def build_model_pipe(self, mode, input):

        # assert isinstance(input,tf.train.Features)
        if mode == 'train':
            # keyWordVector = input['keyWordVector']
            # keyWordVector = tf.reshape(keyWordVector, [self.BatchSize, self.KeyWordNum, self.VecSize], name='KeyWordVector')
            #
            # selWordLabel_r = input['selWordLabel']
            # selLabel = input['selLabel']
            # selLabel = tf.cast(selLabel, tf.float32)
            # selWordLabel = tf.one_hot(selWordLabel_r, depth=self.KeyWordNum)

            wordVector = input['wordVector']
            wordVector = tf.reshape(wordVector, [self.BatchSize, self.ContextLen, self.VecSize], name='WordVector')
            topicSeq = input['topicSeq']
            flagSeq = input['flagSeq']
            topicLabel_r = input['topicLabel']
            flagLabel_r = input['flagLabel']
            wordLabel_r = input['wordLabel']
            topicLabel = tf.one_hot(topicLabel_r, depth=self.TopicNum)
            flagLabel = tf.one_hot(flagLabel_r, depth=self.FlagNum)
            wordLabel = tf.one_hot(wordLabel_r, depth=self.WordNum)


        else:
            keyWordVector = tf.placeholder(tf.float32,
                                           shape=[1, self.KeyWordNum, self.VecSize],
                                           name="Key_Word_Vector")
            wordVector = tf.placeholder(tf.float32,
                                        shape=[1, self.ContextLen, self.VecSize],
                                        name="Context_Vector")
            topicSeq = tf.placeholder(tf.int32,
                                      shape=[1, self.ContextLen],
                                      name="Topic_Sequence")
            flagSeq = tf.placeholder(tf.int32,
                                     shape=[1, self.ContextLen],
                                     name="Flag_Sequence")

        topicVecMap = self.get_variable("Topic_Vec_Map", shape=[self.TopicNum, self.TopicVec], dtype=tf.float32)
        flagVecMap = self.get_variable("Flag_Vec_Map", shape=[self.FlagNum, self.FlagVec], dtype=tf.float32)
        zeros = tf.zeros(shape=[1, self.TopicVec])
        topicVecMap = tf.concat([topicVecMap, zeros], axis=0)
        zeros = tf.zeros(shape=[1, self.FlagVec])
        flagVecMap = tf.concat([flagVecMap, zeros], axis=0)

        topicVec = tf.nn.embedding_lookup(topicVecMap, topicSeq)
        flagVec = tf.nn.embedding_lookup(flagVecMap, flagSeq)
        encAtteWeight = self.get_variable("Encoder_Attention", shape=[self.VecSize, self.ContextVec], dtype=tf.float32)
        selAtteWeight = self.get_variable("Select_Attention", shape=[self.VecSize, self.ContextVec], dtype=tf.float32)

        rawVecSize = self.VecSize + self.TopicVec + self.FlagVec
        rawVecSize *= self.ContextLen

        # 卷积神经网络进行编码的部分
        wordVector = tf.expand_dims(wordVector, -1)
        topicVec = tf.expand_dims(topicVec, -1)
        flagVec = tf.expand_dims(flagVec, -1)

        wordCNN = self.get_cnn_layer(wordVector, name='WordCNN', kernelNum=self.KernelNum)
        topicCNN = self.get_cnn_layer(topicVec, name='TopicCNN', kernelNum=self.KernelNum / 2)
        flagCNN = self.get_cnn_layer(flagVec, name="FlagCNN", kernelNum=self.KernelNum / 2)
        #

        encHiddenWeight = self.get_variable("Encoder_Hidden_0", shape=[self.KernelNum, self.ContextVec], dtype=tf.float32)
        # wordContext = tf.matmul()
        #
        # contextVector = tf.concat([wordVector,topicVec,flagVec],axis=-1,name="Context_Vec_raw")
        contextVector = wordCNN

        # contextVector = tf.reshape(contextVector,[self.BatchSize,-1],name="Flat_context")
        # contextVector = tf.matmul(contextVector,encHiddenWeight)
        #
        # # contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
        # contextVector = tf.nn.tanh(contextVector,name="Context_Vec")

        def attention(q, k, w):
            a = tf.matmul(q, w, transpose_b=True)
            a = tf.expand_dims(a, -1)
            align = tf.matmul(k, a)
            align = tf.nn.softmax(align, axis=1)
            return align, tf.reduce_mean(align * k, 1)

        _, alignKeyWord = attention(contextVector, keyWordVector, encAtteWeight)
        # selVector, _ = attention(contextVector, keyWordVector, selAtteWeight)
        # selVector = tf.squeeze(selVector)
        encVector = tf.concat([alignKeyWord, contextVector], axis=-1, name="Encoder_Vector")
        # encVector = contextVector
        # encVector = tf.nn.l2_normalize(encVector,axis=-1)

        topicH = topicCNN
        # selH = encVector
        flagH = flagCNN
        wordH = encVector

        for i in range(self.HidderLayer):
            topicH = self.get_layer('Topic_Hidden_%d' % i, self.HiddenUnit, topicH)
            # selH = self.get_layer('Sel_Hidden_%d' % i, self.HiddenUnit, selH)
            flagH = self.get_layer('Flag_Hidden_%d' % i, self.HiddenUnit, flagH)
            wordH = self.get_layer('Word_Hidden_%d' % i, self.HiddenUnit, wordH)

        WeightTopic = self.get_variable(name="Weight_Topic", shape=[self.HiddenUnit, self.TopicNum], dtype=tf.float32,
                                        )
        # WeightSel = self.get_variable(name="Weight_Sel", shape=[self.HiddenUnit, 1],
        #                               dtype=tf.float32)
        WeightFlag = self.get_variable(name="Weight_Flag", shape=[self.HiddenUnit, self.FlagNum], dtype=tf.float32,
                                       )
        WeightWord = self.get_variable(name="Weight_Word", shape=[self.HiddenUnit, self.WordNum], dtype=tf.float32,
                                       )

        topicOut = tf.matmul(topicH, WeightTopic, name="Topic_Out")
        # topicOut = tf.nn.dropout(x=topicOut,keep_prob=self.DropoutProb)
        # selOut = tf.matmul(selH, WeightSel, name="Sel_Out")
        # selOut = tf.reshape(selOut, shape=[self.BatchSize])
        # selOut = tf.nn.dropout(x=selOut,keep_prob=self.DropoutProb)
        flagOut = tf.matmul(flagH, WeightFlag, name="Flag_Out")
        # flagOut = tf.nn.dropout(x=flagOut,keep_prob=self.DropoutProb)
        wordOut = tf.matmul(wordH, WeightWord, name="Word_Out")
        # wordOut = tf.nn.dropout(x=wordOut,keep_prob=self.DropoutProb)

        train = tf.no_op()
        loss = tf.no_op()

        ops = {}
        if (mode == 'train'):
            # selRes = tf.nn.sigmoid_cross_entropy_with_logits(logits=selOut, labels=selLabel)
            # topicRes = (1 - selLabel) * tf.nn.softmax_cross_entropy_with_logits_v2(logits=topicOut, labels=topicLabel)
            # flagRes = (1 - selLabel) * tf.nn.softmax_cross_entropy_with_logits_v2(logits=flagOut, labels=flagLabel)
            # selMap = selLabel * tf.nn.softmax_cross_entropy_with_logits_v2(logits=selVector, labels=selWordLabel)
            # wordRes = (1 - selLabel) *tf.nn.softmax_cross_entropy_with_logits_v2(logits=wordOut, labels=wordLabel)
            # selCount = tf.reduce_sum(selLabel)
            # selRes = tf.reduce_mean(selRes, name="Select_Result")
            # topicRes = tf.identity(tf.reduce_sum(topicRes) / (self.BatchSize - selCount + 1), name="Topic_Result")
            # flagRes = tf.identity(tf.reduce_sum(flagRes) / (self.BatchSize - selCount + 1), name="Flag_Result")
            # wordRes = tf.identity(tf.reduce_sum(wordRes) / (self.BatchSize - selCount + 1), name="Word_Result")
            # selMap = tf.identity(tf.reduce_sum(selMap) / (selCount + 1), name="Select_Map_Result")

            topicRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=topicOut, labels=topicLabel)
            flagRes =  tf.nn.softmax_cross_entropy_with_logits_v2(logits=flagOut, labels=flagLabel)
            wordRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=wordOut, labels=wordLabel)
            topicRes = tf.identity(tf.reduce_mean(topicRes) , name="Topic_Result")
            flagRes = tf.identity(tf.reduce_mean(flagRes) , name="Flag_Result")
            wordRes = tf.identity(tf.reduce_mean(wordRes) , name="Word_Result")


            wordGen = tf.argmax(wordOut, axis=-1)
            topicGen = tf.argmax(topicOut, axis=-1)
            flagGen = tf.argmax(flagOut, axis=-1)

            selCount = 1
            wordPrec = tf.reduce_sum(tf.cast(tf.equal(wordGen, wordLabel_r), tf.float32)) / (self.BatchSize - selCount + 1)
            topicPrec = tf.reduce_sum(tf.cast(tf.equal(topicGen, topicLabel_r), tf.float32)) / (
                    self.BatchSize - selCount + 1)
            flagPrec = tf.reduce_sum(tf.cast(tf.equal(flagGen, flagLabel_r), tf.float32)) / (self.BatchSize - selCount + 1)

            tf.summary.scalar('WordPrecision', wordPrec)
            tf.summary.scalar('TopicPrecision', topicPrec)
            tf.summary.scalar('FlagPrecision', flagPrec)

            lossesTensor = [topicRes, flagRes, wordRes]
            # lossesTensor = [selRes, topicRes, flagRes, wordRes, selMap]
            for l in lossesTensor:
                tf.summary.scalar(name=l.name, tensor=tf.reduce_mean(l))

            loss = topicRes + flagRes + wordRes
            omega = tf.add_n(tf.get_collection('l2norm'))
            loss = loss + omega
            tf.summary.scalar('Loss', loss)
            global_step = tf.placeholder(dtype=tf.int32, shape=[], name='Global_Step')
            lr_p = global_step / 200
            lr_tmp = (self.LRDecayRate ** lr_p) * self.LearningRate
            # opt = tf.train.GradientDescentOptimizer(learning_rate=lr_tmp)
            opt = tf.train.AdamOptimizer(learning_rate=lr_tmp)
            grads = opt.compute_gradients(loss)
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)

            grad, var = zip(*grads)
            tf.clip_by_global_norm(grad, self.GlobalNorm)
            grads = zip(grad, var)

            train = opt.apply_gradients(grads)
            ops['GlobalStep'] = global_step
            ops['LearningRate'] = lr_tmp
            ops['WordPrecision'] = wordPrec
            ops['TopicPrecision'] = topicPrec
            ops['FlagPrecision'] = flagPrec
        else:
            # selRes = tf.argmax(selOut, -1)
            topicRes = tf.nn.softmax(topicOut)
            flagRes = tf.nn.softmax(flagOut)
            wordRes = tf.nn.softmax(wordOut)
            # selMap = tf.argmax(selVector)
            # ops['keyWordVector'] = keyWordVector
            ops['wordVector'] = wordVector
            ops['topicSeq'] = topicSeq
            ops['flagSeq'] = flagSeq

            ops['wordRes'] = wordRes
            ops['topicRes'] = topicRes
            ops['flagRes'] = flagRes
            # ops['selRes'] = selRes
            # ops['selMap'] = selMap

        merge = tf.summary.merge_all()
        ops['train'] = train
        ops['merge'] = merge
        ops['loss'] = loss

        return ops

    # def build_model(self,mode):
    #
    #      keyWordVector = tf.placeholder(tf.float32,
    #                                     shape=[self.BatchSize,self.KeyWordNum,self.VecSize],
    #                                     name="Key_Word_Vector")
    #      wordVector = tf.placeholder(tf.float32,
    #                                     shape=[self.BatchSize,self.ContextLen,self.VecSize],
    #                                     name="Context_Vector")
    #      topicSeq = tf.placeholder(tf.int32,
    #                                shape=[self.BatchSize,self.ContextLen],
    #                                name="Topic_Sequence")
    #      flagSeq = tf.placeholder(tf.int32,
    #                                shape=[self.BatchSize,self.ContextLen],
    #                                name="Flag_Sequence")
    #      topicLabel = tf.placeholder(tf.int32,
    #                                shape=[self.BatchSize],
    #                                name="Topic_Label")
    #      flagLabel = tf.placeholder(tf.int32,
    #                                  shape=[self.BatchSize],
    #                                  name="Flag_Label")
    #      wordLabel = tf.placeholder(tf.int32,
    #                                 shape=[self.BatchSize],
    #                                 name="Word_Label")
    #      selLabel = tf.placeholder(tf.float32,
    #                                 shape=[self.BatchSize],
    #                                 name="Sel_Label")
    #      selWordLabel = tf.placeholder(tf.int32,
    #                                shape=[self.BatchSize],
    #                                name="WordSel_Label")
    #
    #      topicLabel = tf.one_hot(topicLabel,depth=self.TopicNum)
    #      flagLabel = tf.one_hot(flagLabel,depth=self.FlagNum)
    #      wordLabel = tf.one_hot(wordLabel,depth=self.WordNum)
    #      selWordLabel = tf.one_hot(selWordLabel,depth=self.KeyWordNum)
    #
    #      topicVecMap = self.get_variable("Topic_Vec_Map",shape=[self.TopicNum,self.TopicVec],dtype=tf.float32,
    #                                   )
    #      flagVecMap = self.get_variable("Flag_Vec_Map",shape=[self.FlagNum,self.FlagVec],dtype=tf.float32,
    #                                   )
    #
    #      topicVec = tf.nn.embedding_lookup(topicVecMap,topicSeq)
    #      flagVec = tf.nn.embedding_lookup(flagVecMap,flagSeq)
    #      rawVecSize = self.VecSize + self.TopicVec+self.FlagVec
    #      rawVecSize *= self.ContextLen
    #      encAtteWeight = self.get_variable("Encoder_Attention",shape=[self.VecSize,self.ContextVec],dtype=tf.float32,
    #                                      )
    #      selAtteWeight = self.get_variable("Select_Attention",shape=[self.VecSize,self.ContextVec],dtype=tf.float32,
    #                                      )
    #      encHiddenWeight = self.get_variable("Encoder_Hidden_0",shape=[rawVecSize,self.ContextVec],dtype=tf.float32,
    #                                        )
    #
    #      contextVector = tf.concat([wordVector,topicVec,flagVec],axis=-1,name="Context_Vec_raw")
    #      contextVector = tf.reshape(contextVector,[self.BatchSize,-1],name="Flat_context")
    #      contextVector = tf.matmul(contextVector,encHiddenWeight)
    #
    #      contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
    #      contextVector = tf.nn.tanh(contextVector,name="Context_Vec")
    #
    #      def attention(q,k,w):
    #          a = tf.matmul(q,w,transpose_b=True)
    #          a = tf.expand_dims(a,-1)
    #          align = tf.matmul(k,a)
    #          align = tf.nn.softmax(align,axis=1)
    #          return align,tf.reduce_mean(align * k,1)
    #
    #      _,alignKeyWord = attention(contextVector,keyWordVector,encAtteWeight)
    #      selVector,_ = attention(contextVector,keyWordVector,selAtteWeight)
    #      selVector = tf.squeeze(selVector)
    #      encVector = tf.concat([alignKeyWord,contextVector],axis=-1,name = "Encoder_Vector")
    #      WeightTopic = self.get_variable(name="Weight_Topic",shape=[self.VecSize+self.ContextVec,self.TopicNum],dtype=tf.float32,
    #                                      )
    #      WeightSel = self.get_variable(name="Weight_Sel", shape=[self.VecSize + self.ContextVec,1],
    #                                      dtype=tf.float32,
    #                                     )
    #      WeightFlag = self.get_variable(name="Weight_Flag",shape=[self.VecSize+self.ContextVec,self.FlagNum],dtype=tf.float32,
    #                                      )
    #      WeightWord = self.get_variable(name="Weight_Word", shape=[self.VecSize + self.ContextVec,self.WordNum], dtype=tf.float32,
    #                                      )
    #
    #      topicOut = tf.matmul(encVector,WeightTopic,name="Topic_Out")
    #      topicOut = tf.nn.dropout(x=topicOut,keep_prob=self.DropoutProb)
    #      selOut = tf.matmul(encVector, WeightSel, name="Sel_Out")
    #      selOut = tf.reshape(selOut,shape=[self.BatchSize])
    #      selOut = tf.nn.dropout(x=selOut,keep_prob=self.DropoutProb)
    #      flagOut = tf.matmul(encVector,WeightFlag,name="Flag_Out")
    #      flagOut = tf.nn.dropout(x=flagOut,keep_prob=self.DropoutProb)
    #      wordOut = tf.matmul(encVector,WeightWord,name="Word_Out")
    #      wordOut = tf.nn.dropout(x=wordOut,keep_prob=self.DropoutProb)
    #
    #      train = tf.no_op()
    #      loss = tf.no_op()
    #
    #      print(selWordLabel)
    #      print(selVector)
    #      if(mode == 'train'):
    #          selRes = tf.nn.sigmoid_cross_entropy_with_logits(logits=selOut,labels=selLabel,name="Select_Result")
    #          topicRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=topicOut,labels=topicLabel,name="Topic_Result")
    #          flagRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flagOut,labels=flagLabel,name="Flag_Result")
    #          wordRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=wordOut,labels=wordLabel,name="Word_Result")
    #          selMap = tf.nn.softmax_cross_entropy_with_logits_v2(logits=selVector,labels=selWordLabel,name="Select_Map_Result")
    #
    #          lossesTensor = [selRes,topicRes,flagRes,wordRes,selMap]
    #          for l in lossesTensor:
    #              tf.summary.scalar(name=l.name,tensor=l)
    #          loss = selRes + (1-tf.sigmoid(selOut))*(topicRes+flagRes+wordRes) +tf.sigmoid(selOut)*(selMap)
    #          loss = tf.reduce_mean(loss)
    #          opt = tf.train.AdamOptimizer(learning_rate=self.LearningRate)
    #          grads = opt.compute_gradients(loss)
    #          for grad,var in grads:
    #              tf.summary.histogram(var.name+'/gradient',grad)
    #
    #          grad,var = zip(*grads)
    #          tf.clip_by_global_norm(grad,self.GlobalNorm)
    #          grads = zip(grad,var)
    #          train = opt.apply_gradients(grads)
    #      merge = tf.summary.merge_all()
    #      ops = {
    #          'keyWordVector':keyWordVector,
    #          'wordVector':wordVector,
    #          'topicSeq':topicSeq,
    #          'flagSeq':flagSeq,
    #          'topicLabel':topicLabel,
    #          'flagLabel':flagLabel,
    #          'wordLabel':wordLabel,
    #          'selLabel':selLabel,
    #          'selWordLabel':selWordLabel,
    #          'train':train,
    #          'loss':loss,
    #          'merge':merge
    #      }
    #      return ops

    #  def build_model_pipe(self,mode,input):
    #      # assert isinstance(input,tf.train.Features)
    #      if mode == 'train':
    #          keyWordVector = input['keyWordVector']
    #          keyWordVector = tf.reshape(keyWordVector,[self.BatchSize,self.KeyWordNum,self.VecSize],name='KeyWordVector')
    #          wordVector = input['wordVector']
    #          wordVector = tf.reshape(wordVector,[self.BatchSize,self.ContextLen,self.VecSize],name='WordVector')
    #          topicSeq = input['topicSeq']
    #          flagSeq = input['flagSeq']
    #          topicLabel_r = input['topicLabel']
    #          flagLabel_r = input['flagLabel']
    #          wordLabel_r = input['wordLabel']
    #          selWordLabel_r = input['selWordLabel']
    #          selLabel = input['selLabel']
    #          selLabel = tf.cast(selLabel,tf.float32)
    #
    #          topicLabel = tf.one_hot(topicLabel_r,depth=self.TopicNum)
    #          flagLabel = tf.one_hot(flagLabel_r,depth=self.FlagNum)
    #          wordLabel = tf.one_hot(wordLabel_r,depth=self.WordNum)
    #          selWordLabel = tf.one_hot(selWordLabel_r,depth=self.KeyWordNum)
    #
    #      else:
    #          keyWordVector = tf.placeholder(tf.float32,
    #                                         shape=[1,self.KeyWordNum,self.VecSize],
    #                                         name="Key_Word_Vector")
    #          wordVector = tf.placeholder(tf.float32,
    #                                      shape=[1,self.ContextLen,self.VecSize],
    #                                      name="Context_Vector")
    #          topicSeq = tf.placeholder(tf.int32,
    #                                    shape=[1,self.ContextLen],
    #                                    name="Topic_Sequence")
    #          flagSeq = tf.placeholder(tf.int32,
    #                                   shape=[1,self.ContextLen],
    #                                   name="Flag_Sequence")
    #
    #
    #      topicVecMap = self.get_variable("Topic_Vec_Map",shape=[self.TopicNum,self.TopicVec],dtype=tf.float32)
    #      flagVecMap = self.get_variable("Flag_Vec_Map",shape=[self.FlagNum,self.FlagVec],dtype=tf.float32)
    #      zeros = tf.zeros(shape=[1,self.TopicVec])
    #      topicVecMap = tf.concat([topicVecMap, zeros], axis=0)
    #      zeros = tf.zeros(shape=[1, self.FlagVec])
    #      flagVecMap = tf.concat([flagVecMap,zeros],axis=0)
    #
    #      topicVec = tf.nn.embedding_lookup(topicVecMap,topicSeq)
    #      flagVec = tf.nn.embedding_lookup(flagVecMap,flagSeq)
    #      encAtteWeight = self.get_variable("Encoder_Attention", shape=[self.VecSize, self.ContextVec], dtype=tf.float32)
    #      selAtteWeight = self.get_variable("Select_Attention", shape=[self.VecSize, self.ContextVec], dtype=tf.float32)
    #
    #      rawVecSize = self.VecSize + self.TopicVec+self.FlagVec
    #      rawVecSize *= self.ContextLen
    #
    #      # 卷积神经网络进行编码的部分
    #      wordVector = tf.expand_dims(wordVector, -1)
    #      topicVec = tf.expand_dims(topicVec,-1)
    #      flagVec = tf.expand_dims(flagVec,-1)
    #
    #      wordCNN = self.get_cnn_layer(wordVector,name='WordCNN',kernelNum=self.KernelNum)
    #      topicCNN = self.get_cnn_layer(topicVec,name='TopicCNN',kernelNum=self.KernelNum/2)
    #      flagCNN = self.get_cnn_layer(flagVec,name="FlagCNN",kernelNum=self.KernelNum/2)
    #      #
    #
    #      encHiddenWeight = self.get_variable("Encoder_Hidden_0",shape=[self.KernelNum,self.ContextVec],dtype=tf.float32)
    #      # wordContext = tf.matmul()
    #      #
    #      # contextVector = tf.concat([wordVector,topicVec,flagVec],axis=-1,name="Context_Vec_raw")
    #      contextVector = wordCNN
    #
    #      # contextVector = tf.reshape(contextVector,[self.BatchSize,-1],name="Flat_context")
    #      # contextVector = tf.matmul(contextVector,encHiddenWeight)
    #      #
    #      # # contextVector = tf.nn.dropout(contextVector,keep_prob=self.DropoutProb)
    #      # contextVector = tf.nn.tanh(contextVector,name="Context_Vec")
    #
    #      def attention(q,k,w):
    #          a = tf.matmul(q,w,transpose_b=True)
    #          a = tf.expand_dims(a,-1)
    #          align = tf.matmul(k,a)
    #          align = tf.nn.softmax(align,axis=1)
    #          return align,tf.reduce_mean(align * k,1)
    #
    #      _,alignKeyWord = attention(contextVector,keyWordVector,encAtteWeight)
    #      selVector,_ = attention(contextVector,keyWordVector,selAtteWeight)
    #      selVector = tf.squeeze(selVector)
    #      encVector = tf.concat([alignKeyWord,contextVector],axis=-1,name = "Encoder_Vector")
    #      # encVector = tf.nn.l2_normalize(encVector,axis=-1)
    #
    #      topicH = topicCNN
    #      selH = encVector
    #      flagH = flagCNN
    #      wordH = encVector
    #
    #      for i in range(self.HidderLayer):
    #          topicH = self.get_layer('Topic_Hidden_%d'%i,self.HiddenUnit,topicH)
    #          selH = self.get_layer('Sel_Hidden_%d'%i,self.HiddenUnit,selH)
    #          flagH = self.get_layer('Flag_Hidden_%d'%i,self.HiddenUnit,flagH)
    #          wordH = self.get_layer('Word_Hidden_%d'%i,self.HiddenUnit,wordH)
    #
    #      WeightTopic = self.get_variable(name="Weight_Topic",shape=[self.HiddenUnit,self.TopicNum],dtype=tf.float32,
    #                                      )
    #      WeightSel = self.get_variable(name="Weight_Sel", shape=[self.HiddenUnit,1],
    #                                      dtype=tf.float32,
    #                                      )
    #      WeightFlag = self.get_variable(name="Weight_Flag",shape=[self.HiddenUnit,self.FlagNum],dtype=tf.float32,
    #                                      )
    #      WeightWord = self.get_variable(name="Weight_Word", shape=[self.HiddenUnit,self.WordNum], dtype=tf.float32,
    #                                      )
    #
    #      topicOut = tf.matmul(topicH,WeightTopic,name="Topic_Out")
    #      # topicOut = tf.nn.dropout(x=topicOut,keep_prob=self.DropoutProb)
    #      selOut = tf.matmul(selH, WeightSel, name="Sel_Out")
    #      selOut = tf.reshape(selOut,shape=[self.BatchSize])
    #      # selOut = tf.nn.dropout(x=selOut,keep_prob=self.DropoutProb)
    #      flagOut = tf.matmul(flagH,WeightFlag,name="Flag_Out")
    #      # flagOut = tf.nn.dropout(x=flagOut,keep_prob=self.DropoutProb)
    #      wordOut = tf.matmul(wordH,WeightWord,name="Word_Out")
    #      # wordOut = tf.nn.dropout(x=wordOut,keep_prob=self.DropoutProb)
    #
    #      train = tf.no_op()
    #      loss = tf.no_op()
    #
    #      ops = {}
    #      if(mode == 'train'):
    #          selRes = tf.nn.sigmoid_cross_entropy_with_logits(logits=selOut,labels=selLabel)
    #          topicRes = (1-selLabel)*tf.nn.softmax_cross_entropy_with_logits_v2(logits=topicOut,labels=topicLabel)
    #          flagRes = (1-selLabel)*tf.nn.softmax_cross_entropy_with_logits_v2(logits=flagOut,labels=flagLabel)
    #          wordRes = tf.nn.softmax_cross_entropy_with_logits_v2(logits=wordOut,labels=wordLabel)
    #          selMap = selLabel*tf.nn.softmax_cross_entropy_with_logits_v2(logits=selVector,labels=selWordLabel)
    #
    #          selCount = tf.reduce_sum(selLabel)
    #          selRes = tf.reduce_mean(selRes,name="Select_Result")
    #          topicRes = tf.identity(tf.reduce_sum(topicRes)/(self.BatchSize-selCount+1),name="Topic_Result")
    #          flagRes = tf.identity(tf.reduce_sum(flagRes)/(self.BatchSize-selCount+1),name="Flag_Result")
    #          wordRes = tf.identity(tf.reduce_sum(wordRes)/(self.BatchSize-selCount+1),name="Word_Result")
    #          selMap = tf.identity(tf.reduce_sum(selMap)/(selCount+1),name="Select_Map_Result")
    #
    #          wordGen = tf.argmax(wordOut,axis=-1)
    #          topicGen = tf.argmax(topicOut,axis=-1)
    #          flagGen = tf.argmax(flagOut,axis=-1)
    #          wordPrec = tf.reduce_sum(tf.cast(tf.equal(wordGen,wordLabel_r),tf.float32))/(self.BatchSize-selCount+1)
    #          topicPrec = tf.reduce_sum(tf.cast(tf.equal(topicGen,topicLabel_r),tf.float32))/(self.BatchSize-selCount+1)
    #          flagPrec = tf.reduce_sum(tf.cast(tf.equal(flagGen,flagLabel_r),tf.float32))/(self.BatchSize-selCount+1)
    #
    #          tf.summary.scalar('WordPrecision',wordPrec)
    #          tf.summary.scalar('TopicPrecision',topicPrec)
    #          tf.summary.scalar('FlagPrecision',flagPrec)
    #
    #          lossesTensor = [selRes,topicRes,flagRes,wordRes,selMap]
    #          for l in lossesTensor:
    #               tf.summary.scalar(name=l.name,tensor=tf.reduce_mean(l))
    #
    #          loss = selMap+topicRes+flagRes+wordRes+selRes
    #          omega = tf.add_n(tf.get_collection('l2norm'))
    #          loss = loss + omega
    #          tf.summary.scalar('Loss',loss)
    #          global_step = tf.placeholder(dtype=tf.int32,shape=[],name='Global_Step')
    #          lr_p = global_step/200
    #          lr_tmp = (self.LRDecayRate**lr_p)*self.LearningRate
    #          # opt = tf.train.GradientDescentOptimizer(learning_rate=lr_tmp)
    #          opt = tf.train.AdamOptimizer(learning_rate=lr_tmp)
    #          grads = opt.compute_gradients(loss)
    #          for grad,var in grads:
    #              tf.summary.histogram(var.name+'/gradient',grad)
    #
    #          grad,var = zip(*grads)
    #          tf.clip_by_global_norm(grad,self.GlobalNorm)
    #          grads = zip(grad,var)
    #
    #          train = opt.apply_gradients(grads)
    #          ops['GlobalStep'] = global_step
    #          ops['LearningRate'] = lr_tmp
    #          ops['WordPrecision'] = wordPrec
    #          ops['TopicPrecision'] = topicPrec
    #          ops['FlagPrecision'] =  flagPrec
    #      else:
    #          selRes = tf.argmax(selOut,-1)
    #          topicRes = tf.nn.softmax(topicOut)
    #          flagRes = tf.nn.softmax(flagOut)
    #          wordRes = tf.nn.softmax(wordOut)
    #          selMap = tf.argmax(selVector)
    #          ops['keyWordVector'] = keyWordVector
    #          ops['wordVector'] = wordVector
    #          ops['topicSeq'] = topicSeq
    #          ops['flagSeq'] = flagSeq
    #
    #          ops['wordRes'] = wordRes
    #          ops['topicRes'] = topicRes
    #          ops['flagRes'] = flagRes
    #          ops['selRes'] = selRes
    #          ops['selMap'] = selMap
    #
    #      merge = tf.summary.merge_all()
    #      ops['train'] = train
    #      ops['merge'] = merge
    #      ops['loss'] = loss
    #
    #      return ops
