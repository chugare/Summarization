import jieba
# import pkuseg
import json
import math
import numpy as np
import re
from data_util.tokenization import tokenization

DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data/NEWS.txt'
E_DATA_PATH = '/home/user/zsm/Summarization/news_data/E_NEWS.txt'



def do_tfidf(doc_g):
    for line in doc_g:
        title,_,source = line.split('#')
        t = Tf_idf()
        tf_v = []
        source_sens = re.split('[，。；;]',source)
        for sen in source_sens:
            sen = sen.split(' ')
            v = t.tf_calc(sen)
            v = np.sum(v)
            tf_v.append((''.join(sen),v))
        tf_v.sort(key=lambda x:x[1],reverse=True)
        res = tf_v[0][0]

        yield ''.join(source.split(' ')),''.join(title.split(' ')),res


class Tf_idf:
    def __init__(self,dic=None,doc_file=None):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.idf = {}
        try:
            _data_file = open('_tfidf_meta.json','r',encoding='utf-8')
            _data_t = json.load(_data_file)
            self.GRAM2N = _data_t['G']
            self.N2GRAM = _data_t['N']
            self.idf = _data_t['I']
        except Exception:
            if dic is None or doc_file is None:
                print('[ERROR] Require data file to initialize')
                return
            self.tokenizer = tokenization(dic,100000)
            self.GRAM2N = self.tokenizer.GRAM2N
            self.N2GRAM = self.tokenizer.N2GRAM
            grams = self.GRAM2N.keys()
            default_idf = [0.0]*100000
            kvs = zip(grams,default_idf)
            self.idf = dict(kvs)

            ga = Tf_idf.read_doc_all(doc_file)
            self.idf_calc(ga)
            _data_file = open('_tfidf_meta.json','w',encoding='utf-8')
            obj = {
                'G':self.GRAM2N,
                'N':self.N2GRAM,
                'I':self.idf
            }
            json.dump(obj,_data_file,ensure_ascii=False)

    def idf_calc(self,doc_gen):
        # doc_data = json.load(self.doc_file)
        doc_num = 0.0
        print('[INFO] Start calc idf')
        for doc in doc_gen:
            tmp_idf = {}
            doc_num+=1
            for w in doc:
                tmp_idf[w] = 1
            for w in tmp_idf:
                if w in self.idf:
                    self.idf[w] += 1
            if int(doc_num) % 100 == 0:
                print('[INFO] %d of doc read'%doc_num)
        print('[INFO] All docs have been read')
        for w in self.idf:
            self.idf[w] = math.log(doc_num/(self.idf[w]+1))
        print('[INFO] All idf value have been calculated')
    def tf_calc(self,sen):
        tf = np.zeros(shape=[len(self.N2GRAM)])
        tf_idf = np.zeros(shape=[len(self.N2GRAM)])
        l = len(sen)
        for word in sen:
            if word in self.GRAM2N:
                tf[self.GRAM2N[word]] = (tf[self.GRAM2N[word]]+1)
                tf_idf[self.GRAM2N[word]] = tf[self.GRAM2N[word]]*self.idf[word]/l

        return  tf_idf
    @staticmethod
    def read_doc_all(fname):
        file_all = open(fname,'r',encoding='utf-8')
        for line in file_all:
            words = line.split(' ')
            yield words

    @staticmethod
    def read_doc_case(fname):
        file_all = open(fname,'r',encoding='utf-8')
        data_all = json.load(file_all)


if __name__ == '__main__':
    Tf_idf(dic=DICT_PATH,doc_file=DATA_PATH)
    # doc_g = open(E_DATA_PATH,'r',encoding='utf-8')
    #
    # tfidf_exp = do_tfidf(doc_g)
    # for real,pred in tfidf_exp:
    #     print("%s --- %s"%(real,pred))

