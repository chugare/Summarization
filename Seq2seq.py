import logging
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from  Tool import  Tf_idf
from DataPipe import DictFreqThreshhold, WordVec


class Meta:
    def __init__(self, **kwargs):
        self.KeyWordNum = 5
        self.VecSize = 300
        self.RNNUnitNum = 800
        self.TopicNum = 30
        self.FlagNum = 60
        self.TopicVec = 10
        self.FlagVec = 20
        self.ContextVec = 400
        self.WordNum = 80000
        self.BatchSize = 256
        self.L2NormValue = 0.02
        self.DropoutProb = 0.7
        self.GlobalNorm = 0.5
        self.LearningRate = 0.01
        self.LRDecayRate = 0.8

        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000

        self.passes = 1
        self.numTopic = 30

        self.Epoch = 10
        self.EpochSize = 100000

        self.ReadNum = 10
        self.LogInterval = 10

        self.EvalCaseNum = 40

        self.MaxSentenceLength = 100
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def get_meta(self):
        return self.__dict__


class Data:
    def __init__(self,**kwargs):
        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000
        self.FlagNum = 60
        self.TopicNum = 30
        self.KeyWordNum = 5
        self.VecSize = 300
        self.MaxSentenceSize = 100
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self.Dict  = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
        self.WordVectorMap = WordVec(**kwargs)
        # lda = LDA.LDA_Train(TaskName = self.TaskName,SourceFile = self.SourceFile,DictName = self.DictName)
        # self.LdaMap = lda.getLda()
        self.DictSize = self.Dict.DictSize
        self.get_word_mat()

        self.TF_IDF = Tf_idf()
    def get_key_word(self,line,num):
        tfvec = self.TF_IDF.tf_calc(line)
        if len(tfvec)<(self.KeyWordNum+2):
            print(tfvec)
            print(line)
            print('')
            return []
        res = self.TF_IDF.get_top_word(tfvec,num)
        return res
    def get_word_mat(self):
        wordNum = self.Dict.DictSize
        flagNum = self.FlagNum
        topicNum = self.TopicNum
        self.TopicWordMat = []
        self.FlagWordMat = []
        for i in range(wordNum):
            ftmp = np.zeros(flagNum)
            ftmp[self.Dict.WF2ID[self.Dict.N2WF[i]]] = 1
            ttmp = np.zeros(topicNum)
            # topic = self.LdaMap[i]
            # if topic == 30:
            #     ttmp = np.ones([topicNum],dtype=np.float32) * (1.0/30.0)
            # else:
            #     ttmp[topic] = 1
            self.TopicWordMat.append(ttmp)
            self.FlagWordMat.append(ftmp)
        self.FlagWordMat =np.array(self.FlagWordMat)
        self.TopicWordMat = np.array(self.TopicWordMat)
    def pipe_data(self):
        sourceFile = open(self.SourceFile, 'r', encoding='utf-8')
        for line in sourceFile:
            line = line.strip()
            words = line.split(' ')
            wordVecList = []
            wordVecList.append(np.random.uniform(-1.0,1.0,[self.VecSize]))
            wordList = []
            wordLength = len(words)
            ref_word = self.get_key_word(words, self.KeyWordNum)
            ref_word = {k: self.WordVectorMap.get_vec(k) for k in ref_word}
            for word in words:
                currentWordId, flag = self.Dict.get_id_flag(word)
                if currentWordId < 0:
                    # 生成单词不从自定义单词表中选择的情况
                    currentWordId = random.randint(0, self.DictSize - 1)
                    # topic = self.LdaMap[currentWordId]
                    # flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    word = self.Dict.N2GRAM[currentWordId]
                wordVec = self.WordVectorMap.get_vec(word)
                wordVecList.append(wordVec)
                wordList.append(currentWordId)
            if len(wordList)>100:
                wordList = wordList[:100]
                wordLength  = 100
            else:
                I = 100-len(wordList)
                for i in range(I):
                    wordList.append(random.randint(0,9999))
                    wordVecList.append(np.zeros([self.VecSize],np.float32))
            wordVecList = wordVecList[:100]
            refMap = {}
            refVector = []
            for i, k in enumerate(ref_word):
                refMap[k] = i
                refVector.append(ref_word[k])
            for i in range(len(refVector), self.KeyWordNum):
                refVector.append(np.zeros([self.VecSize]))
            # print(len(wordVecList))
            # print(len(refVector))2
            yield wordVecList,refVector,wordList,wordLength

    def batch_data(self,batchSize):
        gen = self.pipe_data()
        fp = next(gen)
        data_num = len(fp)
        res = []
        for v in fp:
            res.append([v])
        count = 1
        for t in gen:
            if count >= batchSize:
                count = 0
                yield res
                res = [[] for _ in range(data_num)]
            for i in range(data_num):
                res[i].append(t[i])
            count += 1

class Model:
    def __init__(self,**kwargs):
        self.RNNUnitNum = 300
        self.KeyWordNum = 5
        self.VecSize = 300
        self.HiddenUnit = 800
        self.ContextVec = 400
        self.WordNum = 20000
        self.BatchSize = 64
        self.L2NormValue = 0.02
        self.DropoutProb = 0.7
        self.GlobalNorm = 0.5
        self.LearningRate = 0.01
        self.MaxSentenceLength = 100
        for k,v in kwargs.items():
            self.__setattr__(k,v)
        pass
    def get_attention(self,q,k,name='Attention'):
        # assert q is tf.Tensor
        # assert k is tf.Tensor
        w = self.get_variable(name=name+'_weight',shape = [q.shape[-1],k.shape[-1]],dtype=tf.float32,
                              initializer=tf.glorot_uniform_initializer())
        q = tf.expand_dims(q,1)
        align = tf.tensordot(q,w,[-1,0])*k
        align = tf.reduce_sum(align,axis=-1)
        align = tf.nn.softmax(align)
        align = tf.expand_dims(align,-1)
        v = align*k
        v = tf.reduce_sum(v,1)
        return v
    def get_variable(self,name,shape,dtype,initializer=None):
        if initializer == None:
            initializer = tf.truncated_normal_initializer(stddev=0.2)
        var = tf.get_variable(name,shape,dtype,initializer)
        l2_norm = tf.reduce_mean(var**2) * self.L2NormValue
        tf.add_to_collection('l2norm',l2_norm)
        return var
    def build_model(self,mode):

        SentenceVector = tf.placeholder(dtype=tf.float32,
                                        shape=[self.BatchSize, self.MaxSentenceLength, self.VecSize],
                                        name='Sequence_Vector')
        KeyWordVector = tf.placeholder(dtype=tf.float32, shape=[self.BatchSize, self.KeyWordNum, self.VecSize],
                                       name='Sequence_Vector')
        SentenceLength = tf.placeholder(dtype=tf.int32, shape=[self.BatchSize], name="Sentence_Length")

        weightAtten = self.get_variable(name= 'Attention_weight', shape=[self.RNNUnitNum,self.VecSize], dtype=tf.float32,
                              initializer=tf.glorot_uniform_initializer())
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.RNNUnitNum)
        def loopOpt(i,state,output,maskTA):
            lastWord = SentenceVector[:,i]

            # q = state[0]
            # k = KeyWordVector
            # q = tf.expand_dims(q, 1)
            # align = tf.tensordot(q, weightAtten, [-1, 0]) * k
            # align = tf.reduce_sum(align, axis=-1)
            # align = tf.nn.softmax(align)
            # align = tf.expand_dims(align, -1)
            # v = align * k
            # v = tf.reduce_sum(v, 1)
            # atten = v
            #
            # cellInput = tf.concat([lastWord,atten],axis=-1)
            # mask = tf.cast(tf.greater(SentenceLength,i),dtype=tf.float32)
            #
            # maskTA = maskTA.write(i,mask)

            cellInput = lastWord
            new_output,new_state = cell(cellInput,state)
            q = new_state[0]
            k = KeyWordVector
            q = tf.expand_dims(q, 1)
            align = tf.tensordot(q, weightAtten, [-1, 0]) * k
            align = tf.reduce_sum(align, axis=-1)
            align = tf.nn.softmax(align)
            align = tf.expand_dims(align, -1)
            v = align * k
            v = tf.reduce_sum(v, 1)
            atten = v
            output_vec = tf.concat([new_output,atten],-1)
            mask = tf.cast(tf.greater(SentenceLength,i),dtype=tf.float32)
            maskTA = maskTA.write(i,mask)
            # new_output = new_output*mask
            output = output.write(i,output_vec)
            i = i + 1
            return i,new_state,output,maskTA

        outWeight = self.get_variable('OutLayerWeight',shape=[self.RNNUnitNum+self.VecSize,self.WordNum],dtype=tf.float32,
                                      initializer=tf.glorot_uniform_initializer())


        if mode == 'train':
            WordLabel = tf.placeholder(dtype=tf.int32, shape=[self.BatchSize, self.MaxSentenceLength],
                                       name="Word_Label")

            globalStep = tf.placeholder(dtype=tf.int32, shape=[], name='GlobalStep')
            WordLabelOH = tf.one_hot(WordLabel, self.WordNum)
            initState = cell.zero_state(batch_size=self.BatchSize, dtype=tf.float32)
            outputTA = tf.TensorArray(dtype=tf.float32, size=self.MaxSentenceLength, dynamic_size=False,
                                      clear_after_read=False, tensor_array_name='Output_ta')
            maskTa = tf.TensorArray(dtype=tf.float32, size=self.MaxSentenceLength, dynamic_size=False,
                                    clear_after_read=False, tensor_array_name='Mask_ta')

            i = tf.constant(0, dtype=tf.int32)
            _,finalState,outputTA,maskTa = tf.while_loop(lambda i,*_: i<self.MaxSentenceLength,
                                                         loopOpt,[i,initState,outputTA,maskTa])
            outputTensor = outputTA.stack()
            outputTensor = tf.transpose(outputTensor,[1,0,2])
            maskTensor = maskTa.stack()
            maskTensor = tf.transpose(maskTensor,[1,0])
            res = tf.tensordot(outputTensor,outWeight,[-1,0])
            # res = tf.nn.l2_normalize(res,axis=-1)
            pred = tf.argmax(res,-1)
            pred = tf.cast(pred,tf.int32)
            correct = tf.cast(tf.equal(pred,WordLabel),tf.float32)
            correct = correct * maskTensor
            sample_count = tf.reduce_sum(maskTensor)
            prec = (tf.cast(tf.reduce_sum(correct),dtype=tf.float32))/sample_count
            lr_p = tf.log(tf.cast(globalStep+1, tf.float32))
            lr_tmp = (1/ (lr_p+1)) * self.LearningRate
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=res,labels=WordLabelOH,name='CrossEntropy')
            finalLoss = loss*maskTensor
            omega = tf.add_n(tf.get_collection('l2norm'))
            finalLoss = tf.reduce_sum(finalLoss)/tf.reduce_sum(maskTensor) + omega
            tf.summary.scalar('Loss',finalLoss)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_tmp)
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)
            grad, var = zip(*grads)
            tf.clip_by_global_norm(grad, self.GlobalNorm)
            grads = zip(grad, var)
            merge = tf.summary.merge_all()
            train = optimizer.apply_gradients(grads)
            ops = {
                'SentenceVector':SentenceVector,
                'KeyWordVector':KeyWordVector,
                'WordLabel':WordLabel,
                'SentenceLength':SentenceLength,
                'GlobalStep':globalStep,
                'train':train,
                'loss':finalLoss,
                'maskTensor':maskTensor,
                'precision':prec,
                'merge':merge,
            }
        else:
            initState = cell.zero_state(batch_size=self.BatchSize, dtype=tf.float32)

            i = tf.placeholder(dtype=tf.float32,shape=[])


            state = tf.placeholder(dtype=tf.float32,shape = [self.BatchSize,2,self.RNNUnitNum])
            stateTuple = tf.nn.rnn_cell.LSTMStateTuple(state[:,0],state[:,1])
            runState = tf.cond(tf.equal(i,0), lambda: initState, lambda: stateTuple)

            outputTA = tf.TensorArray(dtype=tf.float32, size=self.MaxSentenceLength, dynamic_size=False,
                                      clear_after_read=False, tensor_array_name='Output_ta')
            maskTa = tf.TensorArray(dtype=tf.float32, size=self.MaxSentenceLength, dynamic_size=False,
                                    clear_after_read=False, tensor_array_name='Mask_ta')

            _,new_state,outputTA,maskTa =  loopOpt(0,runState,outputTA,maskTa)
            outputTensor = outputTA.stack()
            outputTensor = tf.transpose(outputTensor, [1, 0, 2])
            res = tf.tensordot(outputTensor,outWeight,[-1,0])
            # new_output = new_output*mask
            res = tf.squeeze(res)
            resProb = tf.nn.softmax(res)
            ops = {
                'SentenceVector': SentenceVector,
                'KeyWordVector': KeyWordVector,
                'oldState':state,
                'newState':new_state,
                'i':i,
                'resProb':resProb,
            }

        return ops


class Main:
    def __init__(self):
        pass
    def run_train(self,**kwargs):
        TaskName = kwargs['TaskName']
        epochSize = kwargs['EpochSize']
        logInterval = kwargs['LogInterval']
        model = Model(**kwargs)
        ops = model.build_model(mode='train')
        initOp = tf.initialize_all_variables()
        dataPipe = Data(**kwargs)

        epoch = kwargs['Epoch']
        if 'CKP_DIR' not in kwargs:
            kwargs['CKP_DIR'] = 'checkpoint_' + TaskName + '/'

        if 'SUMMARY_DIR' not in kwargs:
            kwargs['SUMMARY_DIR'] = 'summary_' + TaskName + '/'

        checkpoint_dir = os.path.abspath(kwargs['CKP_DIR'])  # meta
        summary_dir = os.path.abspath(kwargs['SUMMARY_DIR'])  # meta
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # data_meta = kwargs['DataMeta']  # meta
        # 模型搭建
        # with tf.device('/cpu:0'):
        # with tf.device('/device:GPU:0'):

        # 训练过程
        saver = tf.train.Saver()
        config = tf.ConfigProto(
            # log_device_placement=True

        )
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # 训练配置，包括参数初始化以及读取检查点

            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            sess.graph.finalize()
            train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
            start_epoch = 0
            global_step = 0
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
                start_epoch += int(checkpoint.split('-')[-1])
                global_step += int(checkpoint.split('-')[-2])
            else:
                sess.run(initOp)
            start_time = time.time()
            # 开始训练

            for i in range(start_epoch, epoch):
                inputPipe = dataPipe.batch_data(kwargs['BatchSize'])
                try:
                    batch_count = 0
                    for v in inputPipe:
                        if batch_count == kwargs['EpochSize']:
                            break
                        try:
                            last_time = time.time()

                            loss, _, prec,merge = sess.run(
                                [ops['loss'], ops['train'],ops['precision'] ,ops['merge']], feed_dict={
                                    ops['GlobalStep']: global_step,
                                    ops['SentenceVector']:v[0],
                                    ops['KeyWordVector']:v[1],
                                    ops['WordLabel']:v[2],
                                    ops['SentenceLength']:v[3],
                                })
                            cur_time = time.time()
                            time_cost = cur_time - last_time
                            total_cost = cur_time - start_time
                            if global_step % max(logInterval, 1) == 0:
                                train_writer.add_summary(merge, global_step)
                                # logger.write_log([global_step/10,loss,total_cost])
                            print('[INFO] Batch %d 训练结果：LOSS=%.2f 准确率=%.2f 用时: %.2f 共计用时 %.2f' % (
                                batch_count, loss,prec, time_cost, total_cost))

                            # print('[INFO] Batch %d'%batch_count)
                            # matplotlib 实现可视化loss
                            batch_count += 1
                            global_step += 1

                        except Exception as e:
                            logging.exception(e)
                            print("[INFO] 因为程序错误停止训练，开始保存模型")
                            saver.save(sess, os.path.join(checkpoint_dir,
                                                          kwargs['TaskName'] + '_summary-' + str(global_step)),
                                       global_step=i)

                    print("[INFO] Epoch %d 结束，现在开始保存模型..." % i)
                    saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(global_step)),
                               global_step=i)
                except KeyboardInterrupt:
                    print("[INFO] 强行停止训练，开始保存模型")
                    saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(global_step)),
                               global_step=i)
                    break

    def run_eval(self,**kwargs):
        TaskName = kwargs['TaskName']
        evalCaseNum = kwargs['EvalCaseNum']

        dataPipe = Data(**kwargs)
        dataProvider = dataPipe.batch_data(1)
        model = Model(**kwargs)

        ops = model.build_model('valid')
        if 'CKP_DIR' not in kwargs:
            kwargs['CKP_DIR'] = 'checkpoint_' + TaskName + '/'

        if 'SUMMARY_DIR' not in kwargs:
            kwargs['SUMMARY_DIR'] = 'summary_' + TaskName + '/'

        checkpoint_dir = os.path.abspath(kwargs['CKP_DIR'])  # meta
        # summary_dir = os.path.abspath(kwargs['SUMMARY_DIR'])  # meta
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        # 模型搭建
        # 训练过程
        saver = tf.train.Saver()
        config = tf.ConfigProto(
            # log_device_placement=True

        )
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # 训练配置，包括参数初始化以及读取检查点

            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            # train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print('[INFO] 从上一次的检查点:\t%s开始继续训练任务' % checkpoint)
            else:
                print('[ERROR] 没有找到任何匹配的checkpoint文件')
            sess.graph.finalize()
            start_time = time.time()
            # 开始训练
            for i in range(evalCaseNum):
                try:
                    last_time = time.time()
                    wordVecList, refVector, wordList, wordLength = next(dataProvider)
                    state = np.zeros([1,2,model.RNNUnitNum])
                    SentenceVector = np.zeros(shape=[model.VecSize],dtype=np.float32)
                    wordList = []
                    for l in range(wordLength[0]):
                        SentenceVector = np.reshape(SentenceVector,[1,1,-1])
                        prob, newState = sess.run([ops['resProb'],ops['newState']],
                                          feed_dict={
                                              ops['SentenceVector']: SentenceVector ,
                                              ops['KeyWordVector']: refVector,
                                              ops['i']: l,
                                              ops['oldState']: state,

                                          })
                        maxID = np.argmax(prob)
                        genWord = dataPipe.Dict.N2GRAM[maxID]
                        SentenceVector = dataPipe.WordVectorMap.get_vec(genWord)
                        state = np.array(newState)
                        state = np.transpose(state,[1,0,2])
                        wordList.append(genWord)

                    print(' '.join(wordList))
                    cur_time = time.time()
                    time_cost = cur_time - last_time
                    total_cost = cur_time - start_time
                    print('[INFO] Sample %d 验证结果：Pre=%.2f  用时: %.2f 共计用时 %.2f' % (
                        i, 0, time_cost, total_cost))

                except KeyboardInterrupt:
                    print("[INFO] 强行停止验证 开始保存结果")

                    break


if __name__ == '__main__':

    args = sys.argv

    if len(args)>1:
        meta = Meta(TaskName='DP_lite', BatchSize=64,
                         ReadNum=-1,
                         WordNum = 10000,
                         LearningRate=float(args[2]),
                         EpochSize = 10000,
                         Epoch = 100,
                         SourceFile='DP_lite.txt',
                         DictName="DP_lite_DICT.txt").get_meta()
        if args[1] == 't':
            Main().run_train(**meta)
        elif args[1] == 'v':
            meta['SourceFile'] = 'E_'+meta['SourceFile']
            meta['BatchSize'] = 1
            meta['MaxSentenceLength'] = 1
            Main().run_eval(**meta)


    else:
        meta = Meta(TaskName = 'DP_lite',BatchSize = 64 ,ReadNum = -1,
                    WordNum = 10000,
                         LearningRate = 0.01,
                    Epoch = 100,
                         SourceFile='DP_lite.txt',
                         DictName = "DP_lite_DICT.txt").get_meta()
        Main().run_train(**meta)