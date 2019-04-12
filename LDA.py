from gensim import corpora,models,similarities
from gensim.models.callbacks import PerplexityMetric,Callback
import sys
import jieba
import numpy as np
import DataPipe
import logging
logging.basicConfig(level=logging.INFO)
class LDA_Train:
    def __init__(self,**kwargs):
        self.passes = 100
        self.numTopic = 30
        self.sourceFile = ""
        self.taskName = "defaultLDA"
        self.dicName = "DICT.txt"
        for k in kwargs:
            self.__setattr__(k,kwargs[k])
    def lda_build(self):
        text = open(self.sourceFile,'r',encoding='utf-8')
        text_set = [line.split(' ') for line in text]
        dic = DataPipe.DictFreqThreshhold(dicName = self.dicName)
        # ddic = corpora.Dictionary(text_set)

        corpus = [dic.doc2bow(line) for line in text_set]
        lda = models.LdaModel(corpus,num_topics=self.numTopic,passes=self.passes)

        lda.save(self.taskName+'_model')
    def read_lda(self):
        lda = models.LdaModel.load(self.taskName+'_model')
        dic = DataPipe.DictFreqThreshhold(dicName = self.dicName)
        for i in range(lda.num_topics):
            tl = lda.get_topic_terms(i)
            res = []
            for w in tl:
                word = dic.get(w[0])
                res.append("%s,%f"%(word,w[1]))
            print(' '.join(res))

if __name__ == '__main__':
    arg = sys.argv
    if arg[1] == '-b':
        ldaInstance = LDA_Train(sourceFile = arg[2],taskName="DP",dicName = 'DP_DICT.txt')
        ldaInstance.lda_build()

