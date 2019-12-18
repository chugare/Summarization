import re
import jieba.posseg as pseg
from data_util.word2vector import WordVec
import json
from util.mysql_utils import dt_write
def TXT2TXT_extract(sourceFile, TaskName, dis_file=None, min_line=50,
                    evalSize=1000, threshold=5):
    sourceFile = open(sourceFile, 'r', encoding='utf-8')
    if dis_file == None:
        dis_file = TaskName + ".txt"
    data_file = open(dis_file, 'w', encoding='utf-8')
    eval_file = open('E_' + dis_file, 'w', encoding='utf-8')

    countFile = 0
    length_map = {}
    dic = {}
    dic_pos = {}

class DatasetBuilder:
    '''
    为了应对不同格式的数据集的创建需求

    数据集创建的基本需求是便利数据来源

    不同的构建方式需要记录不同的数据

    '''
    def __init__(self, source, TaskName, dis_file=None, min_line=50,
                 evalSize=1000, threshold=5):
        self.source = source
        self.TaskName = TaskName
        self.evalSize = evalSize
        self.threshold = threshold
        if dis_file == None:
            dis_file = TaskName + ".txt"

        self.min_line = min_line
        self.dis_file = dis_file
        self.length_map = {}
        self.dic = {}
        self.dic_pos = {}

    def read(self):
        # raise NotImplementedError
        pass

    def cut_without_comma(self, commentLine):
        commentLine = commentLine.replace('\n', '')
        sens = re.split(r"[,、，。；：\n]", commentLine)
        patterns = [
            r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
            r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
            # r"[\s]",
            r"[a-zA-Z《》【】（）\s]+",
        ]
        res = []
        for sen in sens:
            for p in patterns:
                sen = re.sub(p, '', sen)
            if len(sen) < 3:
                continue
            cutres = pseg.lcut(sen)
            for w in cutres:
                wc = w.word
                wf = w.flag
                if wc not in self.dic:
                    self.dic[wc] = 0
                    self.dic_pos[wc] = {}
                self.dic[wc] += 1
                if wf not in self.dic_pos[wc]:
                    self.dic_pos[wc][wf] = 0
                    self.dic_pos[wc][wf] += 1
            lc = len(cutres)
            if lc not in self.length_map:
                self.length_map[lc] = 0
            self.length_map[lc] += 1
            cutres = list(zip(*cutres))
            sen = ' '.join(list(cutres[0]))
            res.append(sen)
            return ' '.join(res)

    def cut_with_comma(self, commentLine):

        patterns = [
            r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
            r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
            # r"[\s]",
            r"[a-zA-Z《》【】（）\s]+",
        ]
        for p in patterns:
            commentLine = re.sub(p, '', commentLine)

            cutres = pseg.lcut(commentLine)
            for w in cutres:
                wc = w.word
                wf = w.flag
                if wc not in self.dic:
                    self.dic[wc] = 0
                    self.dic_pos[wc] = {}
                self.dic[wc] += 1
                if wf not in self.dic_pos[wc]:
                    self.dic_pos[wc][wf] = 0
                self.dic_pos[wc][wf] += 1
            lc = len(cutres)
            if lc not in self.length_map:
                self.length_map[lc] = 0
            self.length_map[lc] += 1
            cutres = list(zip(*cutres))
            return ' '.join(list(cutres[0]))

    ecount = 0

    def cut_with_comma_sen(self, commentLine):

        patterns = [
            r"[（\(]+[一二三四五六七八九十\d]+[\)）]+[，、。．,\s]*",
            r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇]",
            # r"[\s]",
            r"[a-zA-Z《》【】（）\s]+",
        ]
        for p in patterns:
            commentLine = re.sub(p, '', commentLine)
        sens = re.split(r"[,、，。；：\n]", commentLine)
        res = []
        for sen in sens:
            cutres = pseg.lcut(sen)
            for w in cutres:
                wc = w.word
                wf = w.flag
                if wc not in self.dic:
                    self.dic[wc] = 0
                    self.dic_pos[wc] = {}
                self.dic[wc] += 1
                if wf not in self.dic_pos[wc]:
                    self.dic_pos[wc][wf] = 0
                self.dic_pos[wc][wf] += 1
            lc = len(cutres)
            if lc not in self.length_map:
                self.length_map[lc] = 0
            self.length_map[lc] += 1
            cutres = list(zip(*cutres))
            res.append(' '.join(list(cutres[0])))
        return '\n'.join(res)

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



class DPDatasetBuilder(DatasetBuilder):

    def read(self):
        countFile = 0
        for line in self.source:
            line = line.strip()
            # if len(line) != 0:
            #     commentLine += line
            # else:
            #     if len(commentLine)!=0:
            #         try:
            if countFile >= 1000000:
                break
            commentLine = line
            commentLine = commentLine.replace('\n', '')
            if len(commentLine) < self.min_line:
                continue
            countFile += 1
            yield countFile, commentLine


    def build_dataset(self):
        data_file = open(self.dis_file, 'w', encoding='utf-8')
        eval_file = open('E_' + self.dis_file, 'w', encoding='utf-8')
        gen = self.read()
        e_c = 0
        BR1 = BatchWriter(data_file)
        BR2 = BatchWriter(eval_file)
        for i, sentence in gen:
            print("[INFO] Now reading Line : %d " % (i))
            if e_c < self.evalSize:
                e_c += 1
                BR2.write(self.cut_with_comma(sentence))
            else:
                BR1.write(self.cut_with_comma(sentence))
        BR1.close()
        BR2.close()

        dic_file = open(self.TaskName + '_DICT.txt', 'w', encoding='utf-8')
        BRD = BatchWriter(dic_file)
        # pos_file = open('POS.txt','w',encoding='utf-8')
        count = 0
        ULSW = ['\n', '\t', ' ', '']
        for i in ULSW:
            self.dic[i] = 0
        for w in self.dic:
            if self.dic[w] > self.threshold:
                word_type = max(self.dic_pos[w], key=lambda x: self.dic_pos[w][x])
                wordCount = self.dic[w]
                BRD.write("%d %s %s %d" % (count, w, word_type, wordCount))
                count += 1
        BRD.close()
        # pos_file.close()
        print("[INFO] 点评文本读取完毕 共计%d 文本 句子长度统计如下" % count)
        # for kv in enumerate(length_map):
        ll = sorted(self.length_map.keys(), key=lambda x: x)
        for k in ll:
            print("k = %d : %d" % (k, self.length_map[k]))

class LenghtGapDPDataset(DatasetBuilder):
    def __init__(self, sourceFile, TaskName, dis_file=None, min_line=50,
                 evalSize=1000, threshold=5):
        super(LenghtGapDPDataset, self).__init__(sourceFile, TaskName, dis_file, min_line,
                                                 evalSize, threshold)

    def build_dataset(self):
        gen = self.read()
        BRS = []
        for i in range(7):
            fp = open("%d_%d.txt" % (50 + i * 50, 100 + i * 50), 'w', encoding='utf-8')
            BRS.append(BatchWriter(fp))

        for i, sentence in gen:
            if i % 1000 == 0:
                print("[INFO] Now reading Line : %d " % (i))
            k = int(len(sentence) / 50)
            if 0 < k < 7:
                seg_sen = self.cut_with_comma(sentence)
                BRS[k].write(seg_sen)
        for i,BR in enumerate(BRS):
            BR.close()
            print("%d_%d.txt 数据量 %d" % (50 + i * 50, 100 + i * 50, BR.count))


        dic_file = open(self.TaskName + '_DICT.txt', 'w', encoding='utf-8')

        BRD = BatchWriter(dic_file)
        count = 0
        ULSW = ['\n', '\t', ' ', '']
        for i in ULSW:
            self.dic[i] = 0
        for w in self.dic:
            if self.dic[w] > self.threshold:
                word_type = max(self.dic_pos[w], key=lambda x: self.dic_pos[w][x])
                wordCount = self.dic[w]
                BRD.write("%d %s %s %d" % (count, w, word_type, wordCount))
                count += 1
        BRD.close()
        # pos_file.close()
        print("[INFO] 文本读取完毕 共计%d单词 句子长度统计如下" % count)
        # for kv in enumerate(length_map):
        ll = sorted(self.length_map.keys(), key=lambda x: x)
        for k in ll:
            print("k = %d : %d" % (k, self.length_map[k]))

