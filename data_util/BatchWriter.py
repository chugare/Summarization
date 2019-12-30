
class BatchWriter():
    def __init__(self, fp, buffer_size=1000):
        self.fp = fp
        self.buffer = []
        self.buffer_size = buffer_size
        self.count = 0
    def write(self, sen):
        if len(self.buffer) >= self.buffer_size:
            self.fp.write('\n'.join(self.buffer))
            self.buffer = []
            self.count += self.buffer_size
        self.buffer.append(sen)

    def close(self):
        if self.buffer:
            self.fp.write('\n'.join(self.buffer))
            self.count += len(self.buffer)
        self.fp.close()


class BatchWriter_mysql():
    def __init__(self, fp, buffer_size=1000):
        self.fp = fp
        self.buffer = []
        self.buffer_size = buffer_size
        self.count = 0

    def write(self, sen):
        if len(self.buffer) >= self.buffer_size:
            self.fp.write('\n'.join(self.buffer))
            self.buffer = []
            self.count += self.buffer_size
        self.buffer.append(sen)

    def close(self):
        if self.buffer:
            self.fp.write('\n'.join(self.buffer))
            self.count += len(self.buffer)
        self.fp.close()
class SegFileBatchWriter():
    def __init__(self, fp = None,fname = "", buffer_size=10000,data_set_size = 100000):
        self.fname = fname
        self.fp = open(self.fname+'.txt','w',encoding='utf-8')
        self.buffer = []
        self.buffer_size = buffer_size
        self.data_set_size = data_set_size
        self.count = 0
    def write(self, sen):
        if len(self.buffer) >= self.buffer_size:
            self.fp.write('\n'.join(self.buffer))
            self.buffer = []
            self.count += self.buffer_size
            if self.count % self.data_set_size == 0:
                self.fp.close()
                self.fp = open(self.fname+"_"+str(self.count / self.data_set_size)+".txt",'w',encoding='utf-8')
        self.buffer.append(sen)
    def close(self):
        if self.buffer:
            self.fp.write('\n'.join(self.buffer))
            self.count += len(self.buffer)
        self.fp.close()