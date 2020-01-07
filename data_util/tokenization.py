import numpy as np
import json
import jieba

class tokenization:

    def __init__(self, DictName , DictSize = 1000000000):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.wordvec = None
        self.ULSW = ['\n', '\t',' ','']
        self.DictName = DictName
        self.DictSize = DictSize
        self.read_dic()
        self.DictSize = min(len(self.N2GRAM),self.DictSize)

    def read_dic(self):
        try:
            dic_file = open(self.DictName, 'r', encoding='utf-8')
            wordCount = 2
            #
            for line in dic_file:
                wordInfo = line.split(' ')
                if len(wordInfo)<1:
                    continue
                word = wordInfo[1]
                if word in self.ULSW:
                    continue
                # wordIndex = int(wordInfo[0].strip())
                self.GRAM2N[word] = wordCount
                self.N2GRAM[wordCount] = word

                wordCount += 1
                if self.DictSize and wordCount >= self.DictSize:
                    break
            self.GRAM2N['<EOS>'] = 1
            self.N2GRAM[1] = 'SOS'

            self.GRAM2N['<PAD>'] = 0
            self.N2GRAM[0] = 'PAD'

        except FileNotFoundError:
            print('[INFO] 未发现对应的*_DIC.txt文件，需要先生成字典，完毕之后重新运行程序即可')
            print(self.DictName)
            return
        print('[INFO] 字典初始化完毕，共计单词%d个'%len(self.N2GRAM))

    def doc2bow(self,doc):
        if isinstance(doc,list):
            wordSet = doc
        else:
            wordSet = doc.split(' ')
        wordCount = {}
        for w in wordSet:
            if w in self.GRAM2N :
                if w not in wordCount:
                    wordCount[self.GRAM2N[w]] = 0
                wordCount[self.GRAM2N[w]] += 1
        res = sorted(wordCount.items(),key=lambda x:x[1])
        return res


    def tokenize(self,sentence):


        res = []
        for word in sentence:
            if word in self.GRAM2N:
                res.append(self.GRAM2N[word])
            else:
                for w in word:
                    res.append(self.GRAM2N.get(w,0))
        res.append(1)
        return res

    def padding(self, seq, max_length):
        if len(seq) < max_length:
            seq.extend([0] * (max_length - len(seq)))
        else:
            seq = seq[:max_length]
            seq[-1] = 1

        return seq


    def get(self,id):
        if id in self.N2GRAM:
            return self.N2GRAM[id]
        else:
            return ""

    def get_id(self,word):
        if word in self.GRAM2N:
            return self.GRAM2N[word]
        else:
            return -1

    def get_sentence(self, indexArr,cutSize = None):

        res = ''
        for i in range(len(indexArr)):
            if cutSize != None:
                if indexArr[i] > 1:
                    res += '' + (self.N2GRAM[indexArr[i]])
                if len(res)>cutSize or indexArr[i] == 1:
                    break
            else:
                if indexArr[i] != 1:
                    res += '' + (self.N2GRAM[indexArr[i]])
                else:
                    break

        return res

    def get_char_list(self, index_arr):
        res = []
        for i in range(len(index_arr)):
            if index_arr[i] != 1:
                res.append(self.N2GRAM[index_arr[i]])
            else:
                break

        return res



    def Nencoder(self, ec_str):
        grams = jieba.lcut(ec_str)
        ec_vecs = [2]

        for gram in grams:
            if gram in self.GRAM2N:
                ec_vecs.append(self.GRAM2N[gram])
            else:
                # 当词典中没有对应的词时，简单的把单词变成unk符号，抑或是进行进一步的分词？
                ec_vecs.append(0)
        ec_vecs.append(1)
        return np.array(ec_vecs, np.int32)

    def bowencoder(self, ohcode, V):
        res = np.zeros([V], np.int32)
        for c in ohcode:
            res[c] = 1
        return res

    def read_file(self,data_source):
        source = open(data_source, 'r', encoding='utf-8')
        dt = json.load(source)
        for i in dt:
            yield i

    @staticmethod
    def context(title, pos, C):
        res = np.zeros([C], np.int32)
        for i in range(C):
            if pos-i-1<0:
                res[C - i - 1] = 0
            else:
                res[C - i - 1] = title[pos - i - 1]
        return res


