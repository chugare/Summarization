import numpy as np
class RPcore:
    def __init__(self,dict_size,ratio,window_size):
        self.dict_size = dict_size
        self.ratio = ratio
        self.pmap = np.zeros([dict_size],np.float32)
        self.window_size = window_size
        self.buffer = []

    def refresh(self,new_word):

        self.pmap[new_word] += self.ratio
        self.buffer.append(new_word)

        if self.window_size > 0 and self.window_size <len(self.buffer):

            self.pmap[self.buffer[0]] -= self.ratio
            self.buffer.pop(0)

        return self.pmap

    def do(self,sentence,predict):
        new_predict = doRP_simple(self.dict_size,self.ratio,sentence,predict)

        return  new_predict




def doRP_simple(dsize,ratio,sentence,new_predict):
    pmap = np.zeros([dsize],np.float32)
    for w in sentence:
        pmap[w] += ratio
    new_predict -= pmap
    return new_predict

def doRP_window(dsize,ratio,windowsize,sentence,new_predict):
    pmap = np.ones([dsize],np.float32)
    if len(sentence)> windowsize:
        sentence = sentence[-windowsize:]
    for w in sentence:
        pmap[w] += ratio
    new_predict -= pmap
    return new_predict

def doRP_exp(dsize,ratio,drop_rate,sentence,new_predict):
    lg = np.zeros([dsize],np.float32)
    for i,w in enumerate(reversed(sentence)):
        lg[w] += np.power(drop_rate,i)
    pmap = ratio * (lg)
    new_predict -= pmap
    return new_predict
