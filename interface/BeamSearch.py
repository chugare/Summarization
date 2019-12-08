import numpy as np
from data_util import tokenization
class predictor:
    def __init__(self):
        self.end = False
        pass
    def predict(self,source,context,topk = None):
        pass




class Beamsearcher:

    def __init__(self,dataset,topk,predictor):
        self.dataset = dataset
        self.topk = topk
        self.predictor = predictor
        self.buffer = []
        self.tokenization = tokenization.tokenization()

    def do_search(self,max_step):
        for case in self.dataset:
            source = case['source']
            context = [seq[0] for seq in self.buffer]


            self.buffer.append(([],0))
            for i in range(max_step):
                next_topk = self.predictor.predict(source, context)




class TransformerPredictor(predictor):
    def __init__(self,topk = None):
        self.topk = topk

    def predict(self,source,context,topk = None):

        con_len = []
        for seq in context:
            context.append(self.tokenization.padding(seq,max_step))
            con_len.append(len(seq))
        context = np.array(context)
