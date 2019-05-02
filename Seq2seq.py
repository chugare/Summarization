import tensorflow as tf
import random
from DataPipe import DictFreqThreshhold,WordVec
import LDA
import numpy as np

class Model:
    def __init__(self):
        pass
    def build_model(self):

class Main:
    def __init__(self):
        pass

class Data:
    def __init__(self,**kwargs):
        self.SourceFile = 'DP.txt'
        self.TaskName = 'DP'
        self.Name = 'DP_gen'
        self.DictName = "DP_DICT.txt"
        self.DictSize = 80000
        self.FlagNum = 60
        self.TopicNum = 30
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
        self.Dict  = DictFreqThreshhold(DictName = self.DictName,DictSize = self.DictSize)
        self.WordVectorMap = WordVec(**kwargs)
        lda = LDA.LDA_Train(TaskName = self.TaskName,SourceFile = self.SourceFile,DictName = self.DictName)
        self.LdaMap = lda.getLda()
        self.DictSize = self.Dict.DictSize
        self.get_word_mat()
    def get_key_word(self,line,num):
            line = line[:]
            random.shuffle(line)
            num = min(num,len(line))
            return line[:num]
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
    def pipe_data(self):
        sourceFile = open(self.SourceFile, 'r', encoding='utf-8')
        for line in sourceFile:
            line = line.strip()
            words = line.split(' ')
            preWord = [np.zeros([self.VecSize]) for _ in range(self.ContextLen)]
            preTopic = [self.TopicNum for _ in range(self.ContextLen)]
            preFlag = [self.FlagNum for _ in range(self.ContextLen)]
            ref_word = self.get_key_word(words, self.KeyWordNum)
            ref_word = {k: self.WordVectorMap.get_vec(k) for k in ref_word}
            lineBatch = []
            for word in words:
                select = 0
                selectWord = ""
                currentWordId, flag = self.Dict.get_id_flag(word)
                if currentWordId < 0:
                    # 生成单词不从自定义单词表中选择的情况
                    currentWordId = random.randint(0, self.DictSize - 1)
                    topic = self.LdaMap[currentWordId]
                    flag = self.Dict.WF2ID[self.Dict.N2WF[currentWordId]]
                    wordVec = self.WordVectorMap.get_vec(word)
