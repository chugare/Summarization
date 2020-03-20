
import sys
sys.path.append('/home/user/zsm/Summarization')
import tensorflow as tf
import tensorboard as tb
import numpy as np
from data_util.tokenization import tokenization
from util.file_utils import queue_reader
from model.transformer import Transformer,create_look_ahead_mask,create_padding_mask
import time
from interface.ContextTune import CTcore
from baseline.Tf_idf import Tf_idf


from interface.NewsInterface import NewsBeamsearcher,NewsPredictor
# MODEL_PATH = './transformer'
# MODEL_PATH = './tfm_new_off'
# MODEL_PATH = './tfm_1layer_rw'
# MODEL_PATH = './tfm_1layer'
# MODEL_PATH = './tfm_lcsts'
MODEL_PATH = './tfm_lcsts_rw'
DICT_PATH = '/root/zsm/Summarization/news_data_r/NEWS_DICT_R.txt'
# DATA_PATH = '/home/user/zsm/Summarization/news_data'
DATA_PATH = '/home/user/zsm/Summarization/ldata'


NUM_LAYERS=1
D_MODEL=200
NUM_HEAD=8
DFF=512
VOCAB_SIZE=100000
INPUT_LENGTH=150
OUTPUT_LENGTH=40
BATCH_SIZE = 64

R_TF = 1
R_IDF = 0.5
REWEIGHT = True


def generator():
    tokenizer = tokenization(DICT_PATH,DictSize=100000)
    idf_core = Tf_idf(DICT_PATH,DATA_PATH+'/NEWS.txt')
    source_file = queue_reader('lcsts',DATA_PATH)
    for line in source_file:

        example = line.split('#')
        title = example[0]
        desc = example[1]
        content = example[2]
        title = title.split(' ')
        content = content.split(' ')
        source_sequence = tokenizer.tokenize(content)

        source_sequence = tokenizer.padding(source_sequence,INPUT_LENGTH)
        title_sequence = tokenizer.tokenize(title)
        title_sequence = tokenizer.padding(title_sequence,OUTPUT_LENGTH)

        title_sequence.insert(0,2)

        # reweight 操作

        re_weight_map = idf_core.reweight_calc(content,R_IDF,R_TF)
        re_w = [re_weight_map[ti] for ti in title_sequence]
        print(''.join(content))
        for i,v in enumerate(title_sequence):
            if v == 0:
                continue
            print("%s\t%s"%(tokenizer.N2GRAM.get(v,' '),re_w[i]))

        pass

if __name__ == '__main__':
    generator()