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
        self.passes = 1
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
        lda = models.LdaModel(corpus,num_topics=self.numTopic,passes=self.passes,per_word_topics=True)
        dicc = lda.id2word
        size = lda.num_terms
        # dicc = lda.per_word_topics

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
    def getLda(self):
        lda = models.LdaModel.load(self.taskName+'_model')
        assert isinstance(lda,models.LdaModel)
        size = lda.num_terms
        N2Topic = {}
        for i in range(size):
            topicVec = lda.get_term_topics(i, minimum_probability=0.0)
            topicSel = self.numTopic
            if len(topicVec)>0:
                maxTopic = max(topicVec,key=lambda x:x[1])
                topicSel = maxTopic[0]
            N2Topic[i] = topicSel
        return N2Topic
if __name__ == '__main__':
    arg = sys.argv
    if arg[1] == '-b':
        ldaInstance = LDA_Train(sourceFile = arg[2]+'.txt',taskName=arg[2],dicName = arg[2]+'_DICT.txt')
        ldaInstance.lda_build()
    elif arg[1] == '-t':
        ldaInstance = LDA_Train(sourceFile=arg[2] + '.txt', taskName=arg[2], dicName=arg[2] + '_DICT.txt')
        ldaInstance.getLda()
