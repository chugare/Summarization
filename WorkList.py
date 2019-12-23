from data_util import data_pipe
from meta.Meta import Meta
from old_session import Seq2seq_hierarchical_softmax
from util import Tool

meta  = Meta( ReadNum= 80,TaskName = 'DP_s2s_hierarchacal',
                         LearningRate = 0.005,
                         SourceFile='DP_comma.txt',
                         DictName = "DP_comma_DICT.txt").get_meta()
dc = data_pipe.DictFreqThreshhold(**meta)
# dc.HuffmanEncoding(**meta)
dc.getHuffmanDict()
tfidf = Tool.Tf_idf(dic='DP_comma_DICT.txt', doc_file='DP_comma.txt')
meta = Meta(TaskName = 'DP_s2s_hierarchacal',BatchSize = 128 ,ReadNum = 8000,
                         LearningRate = 0.005,
                         SourceFile='DP_comma.txt',
                         DictName = "DP_comma_DICT.txt").get_meta()

main = Seq2seq_hierarchical_softmax.Main()
main.run_train(**meta)