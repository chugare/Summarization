import logging
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from Tool import Tf_idf
from DataPipe import DictFreqThreshhold, WordVec


class Meta:
    def __init__(self, **kwargs):
        self.KeyWordNum = 5
        self.VecSize = 300
        self.RNNUnitNum = 300
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
        self.EpochSize = 1000

        self.ReadNum = 10
        self.LogInterval = 1

        self.EvalCaseNum = 40

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
        self.Dict.HuffmanEncoding()
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
            if len(words) < 20 :
                continue
            wordVecList = []
            # wordVecList.append(np.zeros([self.VecSize],np.float32))
            wordList = []

            ref_word = self.get_key_word(words, self.KeyWordNum)
            if len(ref_word)<self.KeyWordNum:
                continue
            ref_word = {k: self.WordVectorMap.get_vec(k) for k in ref_word}
            wordVecList.append(list(ref_word.values())[0])

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

            # if len(wordList)>100:
            #     wordList = wordList[:100]
            #     wordLength  = 100
            # else:
            #     I = 100-len(wordList)
            #     for i in range(I):
            #         wordList.append(random.randint(0,9999))
            #         wordVecList.append(np.zeros([self.VecSize],np.float32))
            # wordVecList = wordVecList[:100]
            refMap = {}
            refVector = []
            for i, k in enumerate(ref_word):
                refMap[k] = i
                refVector.append(ref_word[k])
            for i in range(len(refVector), self.KeyWordNum):
                refVector.append(np.zeros([self.VecSize]))
            # print(len(wordVecList))
            # print(len(refVector))2
            yield wordVecList,refVector,wordList,len(wordList)
    class Databatchor:
        def __init__(self,generator,**kwargs):
            self.Generator = generator
            self.SampleQueue = {}
            self.SampleReadPos = {}
            self.RNNUnitNum = 800
            self.BatchSize = 64
            for k in kwargs:
                self.__setattr__(k,kwargs[k])
            for i in range(self.BatchSize):
                self.SampleQueue[i] = None
                self.SampleReadPos[i] = 0
        def get_next(self,stateList = None):
            InWordVecList = []
            KeyWordVecList = []
            StateList = [np.zeros([2,self.RNNUnitNum],dtype=np.float32)] *self.BatchSize

            OutWordList = []
            for i in range(self.BatchSize):
                tmpLength = self.SampleReadPos[i] + 1
                if self.SampleQueue[i] is None or \
                        tmpLength >= self.SampleQueue[i][-1] or \
                        stateList is None:
                    self.SampleQueue[i] = next(self.Generator)
                    self.SampleReadPos[i] = 0
                    StateList[i] = np.zeros([2,self.RNNUnitNum],dtype=np.float32)
                    tmpLength = 0
                else:
                    self.SampleReadPos[i] = tmpLength
                    StateList[i] = stateList[:,i,:]
                InWordVecList.append(self.SampleQueue[i][0][tmpLength])
                KeyWordVecList.append(self.SampleQueue[i][1])
                OutWordList.append(self.SampleQueue[i][2][tmpLength])
            StateList = np.transpose(np.array(StateList),[1,0,2])
            return InWordVecList,KeyWordVecList,StateList,OutWordList



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
        self.WordNum = 80000
        self.BatchSize = 64
        self.MaxHuffLength = 19
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
    def build_model(self,mode,huffmanMap,huffmanLabel,huffmanLength):

        InWordVector = tf.placeholder(dtype=tf.float32,
                                        shape=[self.BatchSize, self.VecSize],
                                        name='Word_Vector')
        KeyWordVector = tf.placeholder(dtype=tf.float32, shape=[self.BatchSize, self.KeyWordNum, self.VecSize],
                                       name='Keyword_Vector')

        weightAtten = self.get_variable(name= 'Attention_weight', shape=[self.VecSize,self.VecSize], dtype=tf.float32,
                              initializer=tf.glorot_uniform_initializer())

        WordLabel = tf.placeholder(dtype=tf.int32, shape=[self.BatchSize],
                                   name="Word_Label")

        globalStep = tf.placeholder(dtype=tf.int32, shape=[], name='GlobalStep')
        # index  = tf.placeholder(dtype=tf.float32, shape=[self.BatchSize])

        State = tf.placeholder(dtype=tf.float32, shape=[2,self.BatchSize, self.RNNUnitNum])

        stateTuple = tf.nn.rnn_cell.LSTMStateTuple(State[0], State[1])


        huffmanMap = tf.constant(huffmanMap)
        huffmanLabel = tf.constant(huffmanLabel)
        huffmanLength = tf.constant(huffmanLength)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.RNNUnitNum)



        q = stateTuple[0]
        k = KeyWordVector
        q = tf.expand_dims(q, 1)

        align = tf.tensordot(q, weightAtten, [-1, 0]) * k

        align = tf.reduce_sum(align, axis=-1)
        align = tf.nn.softmax(align)
        tf.summary.histogram('align',align)
        align = tf.expand_dims(align, -1)
        v = align * k
        v = tf.reduce_sum(v, 1)
        print(v)
        atten = v
        # outPut = tf.concat([new_output,atten],-1)


        # cellInput = InWordVector
        cellInput = tf.concat([InWordVector,atten],axis=-1)

        new_output,NewState = cell(cellInput,stateTuple)
        # new_output = new_output*mask
        # q = NewState[0]
        # k = KeyWordVector
        # q = tf.expand_dims(q, 1)
        #
        # align = tf.tensordot(q, weightAtten, [-1, 0]) * k
        #
        # align = tf.reduce_sum(align, axis=-1)
        # align = tf.nn.softmax(align)
        # tf.summary.histogram('align',align)
        # align = tf.expand_dims(align, -1)
        # v = align * k
        # v = tf.reduce_sum(v, 1)
        # print(v)
        # atten = v
        # outPut = tf.concat([new_output,atten],-1)
        # HuffWeight = self.get_variable('HuffmanWeight',shape=[self.WordNum,self.RNNUnitNum+self.VecSize],dtype=tf.float32,
        #                        initializer=tf.glorot_uniform_initializer())

        outPut = new_output

        HuffWeight = self.get_variable('HuffmanWeight',shape=[self.WordNum,self.RNNUnitNum],dtype=tf.float32,
                                      initializer=tf.glorot_uniform_initializer())
        lossTA = tf.TensorArray(dtype=tf.float32,size=self.BatchSize,name='LOSS_TA')

        precMicro = tf.TensorArray(name='PrecMicro', size=self.BatchSize, dtype=tf.int32)
        precMacro = tf.TensorArray(name='PrecMacro', size=self.BatchSize, dtype=tf.int32)
        lsum = tf.constant(0)
        i = tf.constant(0)
        def HierHuffLoop(i,lossTa,pmic,pmac,lSum):
            wordId = WordLabel[i]
            length = huffmanLength[wordId]
            indices = huffmanMap[wordId,:length]
            labels = huffmanLabel[wordId,:length]
            w = tf.gather(HuffWeight, indices)
            out = tf.tensordot(outPut[i], w, [-1, -1])
            result = tf.cast(tf.greater(out, 0.5), tf.int32)
            precAllLevel = tf.cast(tf.equal(result, labels), tf.int32)
            precRes = tf.reduce_prod(precAllLevel)
            precAllLevel = tf.reduce_sum(precAllLevel)
            pmic = pmic.write(i, precAllLevel)
            pmac = pmac.write(i, precRes)
            lab = tf.cast(labels, tf.float32)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=lab)
            loss = tf.reduce_sum(loss)
            lossTa = lossTa.write(i, loss)
            lSum  = lSum + length
            i = i+1
            return i,lossTa,pmic,pmac,lSum
        if mode == 'train':
            _, lossTA, precMicro, precMacro,lsum = tf.while_loop(lambda i, *_: i < self.BatchSize,
                                                            HierHuffLoop, [i, lossTA, precMicro, precMacro,lsum])
            lsum = tf.cast(lsum,dtype=tf.float32)
            precMac = tf.reduce_mean(tf.cast(precMacro.stack(),tf.float32))
            precMic = tf.cast(precMicro.stack(),tf.float32)
            precMic = tf.reduce_sum(precMic)/lsum

            lr_p = tf.log(tf.cast(globalStep + 1, tf.float32))
            lr_tmp = (1 / (lr_p + 1)) * self.LearningRate
            loss_sum = tf.reduce_mean(lossTA.stack())
            omega = tf.add_n(tf.get_collection('l2norm'))

            finalLoss = loss_sum + omega
            tf.summary.scalar('Loss', finalLoss)
            tf.summary.scalar('PrecMac', precMac)
            tf.summary.scalar('PrecMic', precMic)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr_tmp)
            grads = optimizer.compute_gradients(finalLoss)
            for grad, var in grads:
                tf.summary.histogram(var.name + '/gradient', grad)
            # grad, var = zip(*grads)
            # tf.clip_by_global_norm(grad, self.GlobalNorm)
            # grads = zip(grad, var)
            merge = tf.summary.merge_all()
            train = optimizer.apply_gradients(grads)
            ops = {
                'InWordVector': InWordVector,
                'KeyWordVector': KeyWordVector,
                'WordLabel': WordLabel,
                'State':State,
                'GlobalStep': globalStep,
                'train': train,
                'NewState':NewState,
                'loss': finalLoss,
                'precMac': precMac,
                'precMic': precMic,
                'merge': merge,
            }
        else:
            probMap = tf.tensordot(outPut,HuffWeight,[-1,-1])
            probMap = tf.squeeze(probMap)
            ops={
                'InWordVector': InWordVector,
                'KeyWordVector': KeyWordVector,
                'WordLabel': WordLabel,
                'State':State,
                'NewState': NewState,
                'probMap':probMap

            }
        return ops


class Main:
    def __init__(self):
        pass
    def run_train(self,**kwargs):
        TaskName = kwargs['TaskName']
        epochSize = kwargs['EpochSize']
        logInterval = kwargs['LogInterval']
        dataPipe = Data(**kwargs)
        huffTable, huffLabelTable, huffLenTable = dataPipe.Dict.getHuffmanDict()
        model = Model(**kwargs)
        ops = model.build_model('train',huffTable,huffLabelTable,huffLenTable)
        initOp = tf.initialize_all_variables()

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
                start_epoch += int(checkpoint.split('-')[-2])
                global_step += int(checkpoint.split('-')[-1])
            else:
                sess.run(initOp)
            start_time = time.time()
            # 开始训练

            for e in range(start_epoch, epoch):
                inputPipe = dataPipe.pipe_data()
                dataBatchor = Data.Databatchor(inputPipe,**kwargs)

                try:
                    batch_count = 0
                    newState = None
                    for i in range(epochSize):
                        # if batch_count == kwargs['EpochSize']:
                        #     break
                        try:
                            last_time = time.time()
                            v = dataBatchor.get_next(newState)

                            loss, _, outState,precMac,precMic,merge = sess.run(
                                [ops['loss'], ops['train'],ops['NewState'],
                                 ops['precMac'],ops['precMic'],ops['merge']], feed_dict={
                                    ops['GlobalStep']: global_step,
                                    ops['InWordVector']:v[0],
                                    ops['KeyWordVector']:v[1],
                                    ops['State']:v[2],
                                    ops['WordLabel']:v[3],
                                })
                            cur_time = time.time()
                            time_cost = cur_time - last_time
                            total_cost = cur_time - start_time
                            if global_step % max(logInterval, 1) == 0:
                                train_writer.add_summary(merge, global_step)
                                # logger.write_log([global_step/10,loss,total_cost])
                            print('[INFO] Batch %d 训练结果：LOSS=%.2f Macro准确率=%.2f Micro准确率=%.2f 用时: %.2f 共计用时 %.2f' % (
                                batch_count,loss,precMac,precMic,time_cost, total_cost))
                            newState = np.array(outState)
                            # print('[INFO] Batch %d'%batch_count)
                            # matplotlib 实现可视化loss
                            batch_count += 1
                            global_step += 1
                        except StopIteration as exp:
                            logging.exception(exp)
                            print("[INFO] 因为程序错误停止训练，开始保存模型")
                            saver.save(sess, os.path.join(checkpoint_dir,
                                                          kwargs['TaskName'] + '_summary-' + str(e)),
                                       global_step=global_step)
                            break
                        except Exception as exp:
                            logging.exception(exp)
                            print("[INFO] 因为程序错误停止训练，开始保存模型")
                            saver.save(sess, os.path.join(checkpoint_dir,
                                                          kwargs['TaskName'] + '_summary-' + str(e)),
                                       global_step=global_step)
                            return

                    print("[INFO] Epoch %d 结束，现在开始保存模型..." % e)
                    saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(e)),
                               global_step=global_step)
                except KeyboardInterrupt:
                    print("[INFO] 强行停止训练，开始保存模型")
                    saver.save(sess, os.path.join(checkpoint_dir, kwargs['TaskName'] + '_summary-' + str(e)),
                               global_step=global_step)
                    break

    def run_eval(self,**kwargs):
        TaskName = kwargs['TaskName']
        evalCaseNum = kwargs['EvalCaseNum']

        dataPipe = Data(**kwargs)
        huffTable, huffLabelTable, huffLenTable = dataPipe.Dict.getHuffmanDict()
        model = Model(**kwargs)
        ops = model.build_model('valid', huffTable, huffLabelTable, huffLenTable)

        # ops = model.build_model('valid')
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

            inputPipe = dataPipe.pipe_data()
            newState = None

            for i in range(evalCaseNum):
                try:
                    last_time = time.time()

                    wordVecList, refVector, wordList, wordLength = next(inputPipe)
                    state = np.zeros([2,1,model.RNNUnitNum])
                    SentenceVector = np.zeros(shape=[1,model.VecSize],dtype=np.float32)
                    wordList = []
                    refVector = np.expand_dims(refVector,0)

                    for l in range(wordLength):
                        SentenceVector = np.reshape(SentenceVector,[1,-1])
                        newState,prob = sess.run(
                            [ops['NewState'],ops['probMap']], feed_dict={
                                ops['InWordVector']: SentenceVector,
                                ops['KeyWordVector']: refVector,
                                ops['State']: state,
                            })
                        genWord = dataPipe.Dict.read_word_from_Huffman(prob)
                        SentenceVector = dataPipe.WordVectorMap.get_vec(genWord)
                        state = np.array(newState)
                        wordList.append(genWord)
                    resSentence = dataPipe.Dict.get_sentence(wordList)
                    print(resSentence)
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
        meta = Meta(TaskName='DP_lite', BatchSize=256,
                         ReadNum=int(args[1])*1000,
                         WordNum = 10000,
                         LearningRate=float(args[2]),
                         MaxHuffLength = 25,
                         EpochSize = 10000,
                         SourceFile='DP_lite.txt',
                         DictName="DP_lite_DICT.txt").get_meta()
        if args[3] == 't':
            Main().run_train(**meta)
        elif args[3] == 'v':
            meta['SourceFile'] = 'E_'+meta['SourceFile']
            meta['BatchSize'] = 1
            meta['MaxSentenceLength'] = 1
            Main().run_eval(**meta)


    else:
        meta = Meta(TaskName = 'DP_lite',BatchSize = 256 ,ReadNum = -1,
                         LearningRate = 0.01,
                         SourceFile='DP_lite.txt',
                         WordNum = 10000,
                    EpochSize=100000,
                    Epoch = 100,

                    DictName = "DP_lite_DICT.txt").get_meta()
        # dc = DictFreqThreshhold()
        # dc.getHuffmanDict()

        # for i in dp:
        #     sys.stdout.write("\r %d %d %d %d"%(len(i[0]),len(i[1]),len(i[2]),i[3]))
        Main().run_train(**meta)