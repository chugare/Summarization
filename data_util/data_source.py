import re
import jieba.posseg as pseg
from data_util.word2vector import WordVec

def TXT2TXT_extract(sourceFile, TaskName, dis_file=None,min_line = 50,
                    evalSize=1000,threshold = 5):
    sourceFile = open(sourceFile, 'r', encoding='utf-8')
    if dis_file == None:
        dis_file = TaskName + ".txt"
    data_file = open(dis_file, 'w', encoding='utf-8')
    eval_file = open('E_' + dis_file, 'w', encoding='utf-8')

    countFile = 0
    length_map = {}
    dic = {}
    dic_pos = {}





class DatasetBuilder():
    '''
    为了应对不同格式的数据集的创建需求

    数据集创建的基本需求是便利数据来源

    不同的构建方式需要记录不同的数据

    '''
    def __init__(self,sourceFile, TaskName, dis_file=None,min_line = 50,
                    evalSize=1000,threshold = 5):

        self.sourceFile = open(sourceFile, 'r', encoding='utf-8')
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
        pass

    def read(self):
        countFile = 0

        data_file = open(self.dis_file, 'w', encoding='utf-8')
        eval_file = open('E_' + self.dis_file, 'w', encoding='utf-8')

        for line in self.sourceFile:
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
            if (countFile) % 1000 == 0:
                print("[INFO] Now reading Line : %d " % (countFile))
            if self.ecount < 1000:
                self.ecount += 1
                eval_file.write(self.cut_with_comma(commentLine))
                eval_file.write('\n')
            else:
                data_file.write(self.cut_with_comma(commentLine))
                data_file.write('\n')

        dic_file = open(self.TaskName + '_DICT.txt', 'w', encoding='utf-8')
        # pos_file = open('POS.txt','w',encoding='utf-8')
        count = 0
        ULSW = ['\n', '\t', ' ', '']
        for i in ULSW:
            self.dic[i] = 0
        for w in self.dic:
            if self.dic[w] > self.threshold:
                word_type = max(self.dic_pos[w], key=lambda x: self.dic_pos[w][x])
                wordCount = self.dic[w]
                dic_file.write("%d %s %s %d\n" % (count, w, word_type, wordCount))
                count += 1
        dic_file.close()
        # pos_file.close()
        print("[INFO] 点评文本读取完毕 共计%d 文本 句子长度统计如下" % count)
        # for kv in enumerate(length_map):
        ll = sorted(self.length_map.keys(), key=lambda x: x)
        for k in ll:
            print("k = %d : %d" % (k, self.length_map[k]))

    def cut_without_comma(self,commentLine):
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

    def cut_with_comma(self,commentLine):

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

    def cut_with_comma_sen(self,commentLine):

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