from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from model.bert_model import BertConfig,BertModel
import setting
from data_util.dictionary import DictFreqThreshhold
from data_util.word2vector import WordVec
from util.Tool import Tf_idf

flags = tf.flags


FLAGS = flags.FLAGS


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

        self.DictName = self.DictName
        self.SourceFile = setting.DATA_PATH+self.SourceFile
        self.Dict = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
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
            if i not in self.Dict.N2WF:
                continue
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
            wordList = []
            wordLength = len(words)
            ref_words = self.get_key_word(words, self.KeyWordNum)
            if len(ref_words)<self.KeyWordNum:
                continue

            ref_word = self.WordVectorMap.get_vec(ref_words)
            ref_avg_vector = np.average(np.array(list(ref_word.values())),0)
            wordVecList.append(ref_avg_vector) #在第一个正常单词之前输入一个参考的单词，表示文章的第一输入
            word_maps = self.WordVectorMap.get_vec(words)
            for word in words:
                currentWordId, flag = self.Dict.get_id_flag(word)
                if currentWordId < 0:
                    continue
                    # 生成单词不从自定义单词表中选择的情况
                    # currentWordId = random.randint(0, self.DictSize - 1)
                    # # topic = self.LdaMap[currentWordId]
                    # # flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    # word = self.Dict.N2GRAM[currentWordId]
                wordVec = word_maps[word]
                wordVecList.append(wordVec)
                wordList.append(currentWordId)
            if len(wordList)>self.MaxSentenceSize:
                wordList = wordList[:self.MaxSentenceSize]
                wordLength  = self.MaxSentenceSize
            else:
                I = self.MaxSentenceSize-len(wordList)
                for i in range(I):
                    wordList.append(random.randint(0,9999))
                    wordVecList.append(np.zeros([self.VecSize],np.float32))
            wordVecList = wordVecList[:self.MaxSentenceSize]
            refMap = {}
            refVector = []

            # 补全引用文本的数量

            for i, k in enumerate(ref_word):
                refMap[k] = i
                refVector.append(ref_word[k])
            for i in range(len(refVector), self.KeyWordNum):
                refVector.append(np.zeros([self.VecSize]))
            # print(len(wordVecList))
            # print(len(refVector))2
            yield np.array(wordVecList),np.array(refVector),wordList,wordLength,ref_words,line

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
    def __init__(self, **kwargs):
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
        self.BertFile = ""
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        bert_file = open(self.BertFile,'r',encoding='utf-8')
        self.BertConfig = BertConfig.from_json_file(self.BertFile)

    def build_model_fn(self):

        def bert_sum_fn(features, labels, mode, params):
            is_training = (mode == tf.estimator.ModeKeys.TRAIN)


            model = BertModel(
                config=self.BertConfig,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

            final_hidden = model.get_sequence_output()

            final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
            batch_size = final_hidden_shape[0]
            seq_length = final_hidden_shape[1]
            hidden_size = final_hidden_shape[2]


