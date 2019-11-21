import tensorflow as tf

from data_util.tokenization import tokenization
from util.file_utils import queue_reader


def build_input_fn():

    tokenizer = tokenization("./news_data/NEWS_DICT.txt",DictSize=100000)
    source_file = queue_reader("NEWS","/home/user/zsm/Summarization/news_data")
    for line in source_file:
        example = line.split('#')
        title = example[0]
        desc = example[1]
        content = example[2]

        source_sequence = tokenizer.tokenize(content)
        source_sequence = tokenizer.padding(source_sequence,1000)
        title_sequence = tokenizer.tokenize(title)
        title_sequence = tokenizer.padding(title_sequence,100)


    def input_fn():






class AttentionIsAllYourNeed(tf.estimator.Estimator):
