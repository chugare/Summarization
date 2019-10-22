


class DictFreqThreshhold:
    def __init__(self, **kwargs):
        self.GRAM2N = {}
        self.N2GRAM = {}
        self.N2WF = {}
        self.N2FREQ = {}
        self.WF2ID ={}
        self.ID2WF = {}
        self.freq_threshold = 0
        self.wordvec = None
        self.ULSW = ['\n', '\t',' ','']
        self.DictName = 'DP_comma_DICT.txt'
        self.DictSize = 80000
        for k in kwargs:
            self.__setattr__(k,kwargs[k])

        self.read_dic()
        self.DictSize = min(len(self.N2GRAM),self.DictSize)
        self.HuffmanEncoding()
        # print(max(self.N2HUFF, key=lambda k:len(self.N2HUFF[k])))
    def read_dic(self):
        try:
            dic_file = open(self.DictName, 'r', encoding='utf-8')
            wordFlagCount = 0
            wordCount = 0
            freqMode = True
            for line in dic_file:
                wordInfo = line.split(' ')
                if len(wordInfo)<1:
                    continue
                word = wordInfo[1]
                if word in self.ULSW:
                    continue
                wordIndex = int(wordInfo[0].strip())
                wordFlag = wordInfo[2]
                self.GRAM2N[word] = wordIndex
                self.N2GRAM[wordIndex] = word
                if len(wordInfo) > 3 and freqMode:

                    wordFreq = wordInfo[3]
                    try:
                        self.N2FREQ[wordIndex] = int(wordFreq.strip())
                    except Exception as e:
                        print(line)
                        print(wordInfo)
                else:
                    freqMode = False
                self.N2WF[wordIndex] = wordFlag
                if wordFlag not in self.WF2ID:
                    self.WF2ID[wordFlag] = wordFlagCount
                    self.ID2WF[wordFlagCount] = wordFlag
                    wordFlagCount += 1
                wordCount += 1
                if(self.DictSize is not  None ) and wordCount >=self.DictSize:
                    break
        except FileNotFoundError:
            print('[INFO] 未发现对应的*_DIC.txt文件，需要先初始化，初始化完毕之后重新运行程序即可')
            print(self.DictName)
            return
        print('[INFO] 字典初始化完毕，共计单词%d个'%len(self.N2GRAM))
    def dictReformat(self):
        size = len(self.N2GRAM)
        # newDic = []
        tmp_dic = open('tmp_dic.txt','w',encoding='utf-8')
        count = 0
        for k in self.N2GRAM:

            w = self.N2GRAM[k]
            if w in self.ULSW:
                continue
            f = self.N2WF[k]
            freq = self.N2FREQ[k]
            tmp_dic.write('%d %s %s %d\n'%(count,w,f,freq))
            count += 1
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

    def get_id_flag(self,word):
        if word in self.GRAM2N:
            id = self.GRAM2N[word]
            return id,self.WF2ID[self.N2WF[id]]
        else:
            return -1,-1
    def get_sentence(self, indexArr,cutSize = None):

        res = ''
        for i in range(len(indexArr)):
            if cutSize != None:
                if indexArr[i] > 1:
                    res+=(self.N2GRAM[indexArr[i]])
                if len(res)>cutSize:
                    break
            else:
                if indexArr[i] != 1:
                    res+=(self.N2GRAM[indexArr[i]])
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
    def getHuffmanDict(self):
        maxHuffLen = len(self.N2HUFF[max(self.N2HUFF,key=lambda k:len(self.N2HUFF[k]))])
        print('Max Huff Len: %d'%maxHuffLen)
        try:
            meta_file = open('Huffman_Layer.json','r',encoding='utf-8')
            jsdata = json.load(meta_file)

            huffTable = jsdata[0]
            huffLabelTable = jsdata[1]
            huffLenTable = jsdata[2]
            print('[INFO] Huffman Layer Data has been read')
            return huffTable,huffLabelTable,huffLenTable
        except Exception:
            pass
        huffTable = []
        huffLabelTable = []
        huffLenTable = []
        for k in range(self.DictSize):
            # tmphuff = np.zeros(shape=[maxHuffLen],dtype=np.int32)
            tmphuff = [0]*maxHuffLen
            # tmplabel = np.zeros(shape=[maxHuffLen],dtype=np.int32)
            tmplabel = [0]*maxHuffLen
            if str(k) not in self.N2HUFF:
                huffTable.append(tmphuff)
                huffLabelTable.append(tmplabel)
                huffLenTable.append(0)
                continue
            huffman_str = self.N2HUFF[str(k)]
            tlen = len(huffman_str)
            coding = ""
            for i in range(len(huffman_str)):
                if i > 0:
                    coding = huffman_str[:i]
                tmphuff[i] = self.HUFF2LAYER[coding]
                tmplabel[i] = 0 if huffman_str[i] == '0' else 1

            huffTable.append(tmphuff)
            huffLabelTable.append(tmplabel)
            huffLenTable.append(tlen)
        meta_file = open('Huffman_Layer.json','w',encoding='utf-8')
        json.dump([huffTable,huffLabelTable,huffLenTable],meta_file)
        print('[INFO] Huffman Layer Data has been build')

        return huffTable,huffLabelTable,huffLenTable
    def read_word_from_Huffman(self,layersValues):

        encoding = ''
        try:
            while True:
                np = self.HUFF2LAYER[encoding]
                if layersValues[np] >0.5:
                    encoding += '1'
                else:
                    encoding += '0'
        except KeyError:
            if encoding not in self.HUFF2N:
                print('[ERROR] 在字典中没有找到对应的哈夫曼编码 “%s”'%encoding)
                return 0
            else:
                wordId = self.HUFF2N[encoding]
                return wordId
            pass

    def HuffmanEncoding(self,forceBuild = False):
        class HuffmanNode:
            def __init__(self,val = None,word = None):
                self.right = None
                self.left = None
                self.value = val
                self.word = word
                self.huffman = ''
        Nodes = [HuffmanNode(self.N2FREQ[k],k) for k in self.N2FREQ]
        if not forceBuild:
            try:
                meta_file = open('Huffman_dic.json','r',encoding='utf-8')
                self.N2HUFF,self.HUFF2N,self.HUFF2LAYER,self.LAYER2HUFF = json.load(meta_file)
                print('[INFO] Huffman dictionary has been readed')
                return
            except Exception:
                pass
        if len(Nodes) < 1:
            return

        while len(Nodes) > 1:
            Nodes.sort(key=lambda node: node.value, reverse=True)
            nv = Nodes[-1].value + Nodes[-2].value
            tmpNode = HuffmanNode(nv)
            tmpNode.left = Nodes[-2]
            tmpNode.right = Nodes[-1]
            Nodes.pop(-1)
            Nodes.pop(-1)
            Nodes.append(tmpNode)
        self.N2HUFF = {}
        self.HUFF2N = {}
        rootNode = Nodes[0]
        NodeQ = [rootNode]
        c = 0
        self.HUFF2LAYER = {}
        self.LAYER2HUFF = {}

        while len(NodeQ) > 0:
            tmpNode = NodeQ[0]

            NodeQ.pop(0)
            if c %1000==0:
                print('[INFO] Huffman Build %d'%c)
            if tmpNode.word is not None:
                self.N2HUFF[str(tmpNode.word)] = tmpNode.huffman
                self.HUFF2N[tmpNode.huffman] = tmpNode.word
                continue
            self.HUFF2LAYER[tmpNode.huffman] = c
            self.LAYER2HUFF[str(c)] = tmpNode.huffman
            if tmpNode.left is not None:
                tmpNode.left.huffman = tmpNode.huffman + '0'
                NodeQ.append(tmpNode.left)
            if tmpNode.right is not None:
                tmpNode.right.huffman = tmpNode.huffman + '1'
                NodeQ.append(tmpNode.right)
            c+= 1
        meta_file = open('Huffman_dic.json', 'w', encoding='utf-8')
        json.dump([self.N2HUFF,self.HUFF2N,self.HUFF2LAYER,self.LAYER2HUFF],meta_file)


        # for k in self.N2HUFF:
        #     print("%s %d %s"%(self.N2GRAM[k],self.N2FREQ[k],self.N2HUFF[k]))
