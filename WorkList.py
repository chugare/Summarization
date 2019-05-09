from DataPipe import DataPipe
from Meta import  Meta
import Tool,DataPipe,Seq2seq_hierarchical_softmax

Tool.TXT2TXT_extract('../DP/rating_2.txt',"DP_comma",testCase = -1)
meta  = Meta( ReadNum= 800000 ).get_meta()
dc = DataPipe.DictFreqThreshhold(**meta)
dc.HuffmanEncoding(**meta)
dc.getHuffmanDict()
meta = Meta(TaskName = 'DP_s2s_hierarchacal',BatchSize = 128 ,ReadNum = 800000,
                         LearningRate = 0.01,
                         SourceFile='DP_comma.txt',
                         DictName = "DP_comma_DICT.txt").get_meta()

main = Seq2seq_hierarchical_softmax.Main()
main.run_train(**meta)