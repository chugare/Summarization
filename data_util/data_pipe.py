# -*- coding: utf-8 -*-
#   Project name : Evi-Fact
#   Edit with PyCharm
#   Create by simengzhao at 2018/7/30 下午6:53
#   南京大学软件学院 Nanjing University Software Institute
#
#   这个文件用于生成训练所需要的数据，主要包括以下几个方法
#   1. 从文件中读取，生成原始文本与摘要文本的序列
#   2. 将文本序列表示成字典序列
#   3. 生成batch数据
#   4. 去除停用词

import jieba
import numpy as np
import random
import tensorflow as tf
import re
import json
import sys
import os
from model import LDA
from meta.Meta import  Meta


class Reader:
    def __init__(self):
        pass

class DataPipe:

    def __init__(self,**kwargs):
        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self.Dict  = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
        self.WordVectorMap = WordVec(**kwargs)
        lda = LDA.LDA_Train(TaskName = self.TaskName, SourceFile = self.SourceFile, DictName = self.DictName)
        self.LdaMap = lda.getLda()
        self.DictSize = self.Dict.DictSize
        self.get_word_mat()

    def get_key_word(self,line,num):
            line = line[:]
            random.shuffle(line)
            num = min(num,len(line))
            return line[:num]


    def pipe_data(self,**kwargs):

        try:
            contentLen = kwargs['ContextLen']
            vecSize = kwargs['VecSize']
            keyWordNum = kwargs['KeyWordNum']
            topicNum = kwargs['TopicNum']
            flagNum = kwargs['FlagNum']
        except KeyError as e:
            print(str(e))
            return
        sourceFile = open(self.SourceFile,'r',encoding='utf-8')
        for line in sourceFile:
            line  = line.strip()
            words = line.split(' ')
            preWord = [np.zeros([vecSize]) for _ in range(contentLen)]
            preTopic = [topicNum for _ in range(contentLen)]
            preFlag = [flagNum for _ in range(contentLen)]
            ref_word = self.get_key_word(words,keyWordNum)
            ref_word = {k:self.WordVectorMap.get_vec(k) for k in ref_word}
            lineBatch = []
            for word in words:
                select = 0
                selectWord = ""
                currentWordId,flag = self.Dict.get_id_flag(word)
                if currentWordId <0:
                    # 生成单词不从自定义单词表中选择的情况
                    currentWordId = random.randint(0,self.DictSize-1)
                    topic = self.LdaMap[currentWordId]
                    flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    wordVec = self.WordVectorMap.get_vec(word)

                else:
                    topic = self.LdaMap[currentWordId]
                    wordVec = self.WordVectorMap.get_vec(word)

                lineBatch.append((preWord,preTopic,preFlag,topic,flag,currentWordId,select,selectWord))

                preWord = preWord[1:]
                preWord.append(wordVec)
                preFlag = preFlag[1:]
                preFlag.append(flag)
                preTopic = preTopic[1:]
                preTopic.append(topic)

            refMap = {}
            refVector = []
            for i,k in enumerate(ref_word):
                refMap[k] = i
                refVector.append(ref_word[k])
            for i in range(len(refVector),keyWordNum):
                refVector.append(np.zeros([vecSize]))
            for preWord,preTopic,preFlag,topic,flag,currentWordId,select,selectWord in lineBatch:

                # if select > 0:
                #     selectWord = refMap[selectWord]
                # else:
                #     selectWord = keyWordNum
                yield {
                    'wordVector':np.array(preWord,dtype=np.float32),
                    'topicSeq':np.array(preTopic,dtype=np.int64),
                    'flagSeq':np.array(preFlag,dtype=np.int64),
                    'keyWordVector':np.array(refVector,dtype=np.float32),
                    'topicLabel':topic,
                    'flagLabel':flag,
                    'wordLabel':currentWordId,
                    # 'selLabel':select,
                    # 'selWordLabel':selectWord,
                }
        pass

    def write_TFRecord(self,meta,million):

        metaFile = open('meta_tfrecord.json', 'w', encoding='utf-8')
        json.dump(meta, metaFile, ensure_ascii=False)
        metaFile.close()
        gen = self.pipe_data(**meta)
        CountK = 0
        CountM = 0
        KCount = 0
        writer = tf.python_io.TFRecordWriter(self.Name + '-%d.tfrecord'%CountM)
        for v in gen:

            example = self.get_feature(**v)
            writer.write(example.SerializeToString())
            KCount += 1
            if KCount %1000 == 0:
                CountK += 1
                KCount = 0

                print("[INFO] %d K Samples read to record "%CountK)
                if CountK%1000 == 0:
                    CountM+=1
                    print("[INFO] %d M Samples has been read, Writing to record " % CountM)
                    writer.close()
                    if CountM == million:
                        break
                    writer = tf.python_io.TFRecordWriter(self.Name + '-%d.tfrecord'%CountM)

    def read_TFRecord(self,BATCH_SIZE):

        metaFile = open('meta_tfrecord.json','r',encoding='utf-8')
        meta = json.load(metaFile)
        try:
            contentLen = meta['ContextLen']
            vecSize = meta['VecSize']
            keyWordNum = meta['KeyWordNum']

        except KeyError as e:
            print(str(e))
            return
        files = os.listdir('.')
        recordList = []
        for f in files:
            if f.endswith('.tfrecord'):
                recordList.append(f)

        fileQueue = tf.train.string_input_producer(recordList)
        recordReader = tf.TFRecordReader()
        i,serializeExample = recordReader.read(fileQueue)
        features = tf.parse_single_example(serializeExample,features={
            'wordVector': tf.FixedLenFeature(shape=[contentLen*vecSize],dtype=tf.float32),
            'topicSeq': tf.FixedLenFeature(shape=[contentLen],dtype=tf.int64),
            'flagSeq': tf.FixedLenFeature(shape=[contentLen],dtype=tf.int64),
            'keyWordVector': tf.FixedLenFeature(shape=[keyWordNum*vecSize],dtype=tf.float32),
            'topicLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            'flagLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            'wordLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            # 'selLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
            # 'selWordLabel': tf.FixedLenFeature(shape=[],dtype=tf.int64),
        })
        # batchData = tf.train.batch(features,batch_size=BATCH_SIZE,num_threads=4,capacity=1000)
        batchData = tf.train.shuffle_batch(features,batch_size=BATCH_SIZE,
                                           capacity=20000,num_threads=4,
                                           min_after_dequeue=10000)
        return batchData

    def get_feature(self,**kwargs):
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
            topic = self.LdaMap[i]
            if topic == 30:
                ttmp = np.ones([topicNum],dtype=np.float32) * (1.0/30.0)
            else:
                ttmp[topic] = 1
            self.TopicWordMat.append(ttmp)
            self.FlagWordMat.append(ftmp)
        self.FlagWordMat =np.array(self.FlagWordMat)
        self.TopicWordMat = np.array(self.TopicWordMat)

    def _get_prob(self,PW,PF,PT):
        PW_T = np.matmul(self.TopicWordMat,np.transpose(PT))
        PW_F = np.matmul(self.FlagWordMat,np.transpose(PF))
        PW_T = np.squeeze(PW_T)
        PW_F = np.squeeze(PW_F)
        PW = np.squeeze(PW)
        disSize = min(PW.shape[0],PW_F.shape[0])
        PW = PW[:disSize]
        PW_WT = PW * PW_T
        PW_WF = PW * PW_F

        PW_WTF = PW_WT * PW_F
        return PW,PW_WT,PW_WF,PW_WTF

    def get_prob_result(self,PW,PF,PT):
        PW, PW_WT, PW_WF, PW_WTF = self._get_prob(PW,PF,PT)
        IdPW_WTF = np.argmax(PW)
        resWord = self.Dict.N2GRAM[IdPW_WTF]
        resFlag = self.Dict.N2WF[IdPW_WTF]
        resTopic = self.LdaMap[IdPW_WTF]
        return resWord,resFlag,resTopic

    def get_input_data(self,wordSeq=None,topicSeq=None, flagSeq=None, newWord = None):
        if topicSeq is None and flagSeq is None and wordSeq is None :
            wordSeq = [np.zeros(self.VecSize) for _ in range(self.ContextLen)]
            topicSeq = [self.TopicNum for _ in range(self.ContextLen)]
            flagSeq = [self.FlagNum for _ in range(self.ContextLen)]
        else:
            try:
                wordSeq = wordSeq[1:]
                topicSeq = topicSeq[1:]
                flagSeq = flagSeq[1:]
                newWordID = self.Dict.GRAM2N[newWord]
                nVector = self.WordVectorMap.get_vec(newWord)
                nFlag = self.Dict.WF2ID[self.Dict.N2WF[newWordID]]
                nTopic = self.LdaMap[newWordID]
                wordSeq.append(nVector)
                topicSeq.append(nTopic)
                flagSeq.append(nFlag)

            except KeyError as e:
                print(str(e))
        return wordSeq,topicSeq,flagSeq


    def get_next_context(self,preWordSeq,topicSeq,flagSeq,lastWord,lastSelected):
        preWordSeq = preWordSeq[1:]
        id,flag = self.Dict.get_id_flag(lastWord)
        lastWordVector = self.WordVectorMap.get_vec(id)
        preWordSeq.append(lastWordVector)
        topic = self.LdaMap.get(id)
        topicSeq = topicSeq[1:]
        flagSeq = flagSeq[1:]
        topicSeq.append(topic)
        flagSeq.append(flag)

    def get_init_context(self):
        preWordSeq = [np.zeros([self.VecSize]) for _ in range(self.ContextLen)]
        topicSeq = [self.TopicNum  for _ in range(self.ContextLen)]
        flagSeq = [self.FlagNum for _ in range(self.ContextLen)]
        return preWordSeq,topicSeq,flagSeq

    def pipe_data_for_eval(self,**kwargs):
        try:
            vecSize = self.VecSize
            keyWordNum = self.KeyWordNum
            topicNum = self.TopicNum
            flagNum = self.FlagNum
        except KeyError as e:
            print(str(e))
            return
        # try:
        sourceFile = open(self.SourceFile,'r',encoding='utf-8')
        for line in sourceFile:
            line  = line.strip()
            words = line.split(' ')
            topicSeq = []
            ref_word = self.get_key_word(words,keyWordNum)
            ref_word = {k:self.WordVectorMap.get_vec(k) for k in ref_word}
            flagSeq =[]
            for word in words:
                topic = topicNum
                currentWordId,flag = self.Dict.get_id_flag(word)
                if  currentWordId <0:
                    currentWordId = random.randint(0, self.DictSize - 1)
                    topic = self.LdaMap[currentWordId]
                    flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    wordVec = self.WordVectorMap.get_vec(word)

                topic = self.LdaMap[currentWordId]
                topicSeq.append(topic)
                flagSeq.append(flag)
            refMap = {}
            refVector = []
            for i,k in enumerate(ref_word):
                refMap[i] = k
                refVector.append(ref_word[k])
            for i in range(len(refVector),keyWordNum):
                refVector.append(np.zeros([vecSize]))
            ref_word = ref_word.keys()
            yield words,flagSeq,topicSeq,refMap,refVector,ref_word
        # except Exception as e:
        #     print(str(e))




