from gensim import corpora,models,similarities
from gensim.models.callbacks import PerplexityMetric,Callback
import sys
import jieba
import numpy as np
import DataPipe
import logging
logging.basicConfig(level=logging.INFO)
def lda_build(source):
    text = open(source,'r',encoding='utf-8')
    text_set = [line.split(' ') for line in text]
    dic = DataPipe.DictFreqThreshhold()
    # ddic = corpora.Dictionary(text_set)

    corpus = [dic.doc2bow(line) for line in text_set]
    lda = models.LdaModel(corpus,num_topics=30,passes=100)
    name = source.split('.')[0]
    lda.save(name+'_model')
def read_lda(name):
    lda = models.LdaModel.load(name)
    dic = DataPipe.DictFreqThreshhold()
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
        lda_build(arg[2])

