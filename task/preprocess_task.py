from data_util.word2vector import WordVec
from data_util.data_source import TXT2TXT_extract
# w = WordVec()
# w.init_wv_redis()
# TXT2TXT_extract("/home/user/zsm/data/rating_2.txt",'DP')
from util.Tool import Tf_idf

Tf_idf('/home/user/zsm/Summarization/data/DP_DICT.txt','/home/user/zsm/Summarization/data/DP.txt')