import numpy as np
import re
import math
from data_util.tokenization import tokenization
DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
DATA_PATH = '/home/user/zsm/Summarization/news_data/NEWS.txt'
E_DATA_PATH = '/home/user/zsm/Summarization/news_data/E_NEWS.txt'



def simility(sen1,sen2):
    s_c = 0
    for w1 in sen1:
        for w2 in sen2:
            if w1 == w2:
                s_c += 1
    try:
        res = float(s_c)/(math.log(len(sen1)+1)+math.log(len(sen2)+1)+0.1)
        return  res

    except ZeroDivisionError:
        print(sen1)
        print(sen2)


def  textRank_s(source):
    source_sens_raw = re.split('[，。；;]',source)
    source_sens = [''.join(s.split(' ')) for s in source_sens_raw]
    source_size = len(source_sens)
    V = np.ones([source_size], np.float32)
    E = np.zeros([source_size, source_size], np.float32)

    d = 0.85
    for i in range(source_size - 1):
        for j in range(i + 1, source_size):
            sim = simility(source_sens[i], source_sens[j])
            E[i][j] = sim
            E[j][i] = sim
    W_o = np.sum(E, 1)

    def V_iter():
        V_t = np.ones([source_size], np.float32)
        for i in range(source_size):
            sum1 = 0.0
            for j in range(source_size):
                sum1 += E[j][i] * V[j] / (W_o[j] + 0.001)
            V_t[i] = (1 - d) + d * sum1
        return V_t

    def loss(V, V_t):
        loss_v = 0
        for i in range(len(V)):
            loss_v += math.fabs(V[i] - V_t[i])
        return loss_v

    for i in range(100):
        V_t = V_iter()
        loss_v = loss(V, V_t)
        if loss_v < 1e-7:
            V = V_t
            break
        V = V_t
    V_r = V
    V = []
    for i in range(len(V_r)):
        V.append((i, V_r[i]))

    V.sort(key=lambda x: x[1], reverse=True)


    res = source_sens[V[0][0]]
    return  res



def textRank(doc_g):


    for line in doc_g:
        title,_,source = line.split('#')
        res = textRank_s(source)

        yield ''.join(source.split(' ')),''.join(title.split(' ')),res

if __name__ == '__main__':
    # doc_g = open(E_DATA_PATH,'r',encoding='utf-8')
    #
    # txtrank_exp = textRank(doc_g)
    # for source,real,pred in txtrank_exp:
    #     print("%s --- %s"%(real,pred))
    #
    # 用text rank 构建数据集
    doc_g = open(DATA_PATH,'r',encoding='utf-8')
