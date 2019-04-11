from gensim import corpora,models,similarities
from gensim.models.callbacks import PerplexityMetric,Callback
import sys
import jieba
import numpy as np
def lda_build(source):
    text = open(source,'r',encoding='utf-8')
    text_set = [line.split(' ') for line in text]
    dic = corpora.Dictionary(text_set)
    corpus = [dic.doc2bow(line) for line in text_set]
    dic.save('lda_dic')
    class mycallback(Callback):
        def __init__(self):
            self.EPOCH = 0
        def on_epoch_end(self, epoch, topics=None):
    lda = models.LdaModel(corpus,num_topics=30)
    name = source.split('.')[0]
    lda.save(name+'_model')
def read_lda(name):
    lda = models.LdaModel.load(name)
    dic = corpora.Dictionary.load('lda_dic')
    for i in range(lda.num_topics):
        tl = lda.get_topic_terms(i)
        res = []
        for w in tl:
            word = dic.get(w[0])
            res.append("%s,%f"%(word,w[1]))
        print(' '.join(res))

lda_build('DOC.txt')
read_lda('LDA_model')


