import numpy as np
class CTcore:
    def __init__(self,dict_size,ratio):
        self.ratio = ratio
        self.dict_size = dict_size
        pass

    def init(self,source):

        self.tune_mat = np.ones([self.dict_size],np.float32)
        for w in source:
            self.tune_mat[w] = self.ratio

    def do(self,pred,sentence):

        return pred * self.tune_mat

