import numpy as np
from data_util import tokenization
import tensorflow as tf
import  threading,queue
class Predictor:
    def __init__(self):
        self.end = False
        pass
    def predict(self,source,context,conlen,topk = None):
        pass

class Beamsearcher:

    def __init__(self,dataset,tokenizer,topk,predictor,max_count = 10):
        self.dataset = dataset
        self.topk = topk
        self.predictor = predictor
        self.buffer = [] # 保存当前生成内容
        self.gen_len = 0 # 保存生成长度
        self.next_topk = queue.Queue(1)
        self.tokenizer = tokenizer
        self.gen_result = []
        self.max_count = max_count


    def do_search(self,max_step):
        for case in self.dataset:
            source = case
            source_input = [source[:] for _ in range(max_step)]
            self.buffer = []

            self.buffer.append(([],0))
            for i in range(max_step):

                context = []
                # con_len = []

                for seq in self.buffer:
                    # con_len.append(len(seq[0]))
                    context.append(self.tokenizer.padding(seq[0][:],max_step))

                context = np.array(context)

                next_topk = self.predictor.predict(source_input[:context.shape[0]], context, self.gen_len,self.topk)
                tmp_buffer = []

                for i, val in enumerate(self.buffer):
                    candidate,score = val
                    for n in next_topk[i]:
                        ts = next_topk[i][n]
                        tc = candidate[:]
                        tc.append(n)
                        tmp_buffer.append((tc,score+ts))

                sorted(tmp_buffer,key=lambda x:x[1])
                tmp_buffer = tmp_buffer[:self.topk]
                self.buffer = tmp_buffer
                self.gen_len += 1



    # def do_search_mt(self,max_step):
    #     buffer_lock = threading.RLock()
    #     result_lock = threading.RLock()
    #
    #     def fill_data(searcher):
    #         buffer_lock.acquire(timeout=20)
    #         if searcher.buffer









